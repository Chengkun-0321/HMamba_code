import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 
import os
import multiprocessing
from multiprocessing import Pool
from tensorflow import keras
#import util2 as u   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,LeakyReLU,MaxPool2D,Dense,Flatten


# ---------------------------
# 定義模型參數
# ---------------------------
@dataclass
class ModelArgs:
    model_input_dims: int = 128   # 每個 token 投影後的維度
    model_states: int = 32        # 與原架構相容，此處未實際使用
    num_layers: int = 5           # 可供擴充使用，此範例中固定為 5
    dropout_rate: float = 0.2     # (此範例中未使用，可自行添加)
    num_classes: int = 1          # 輸出單位數，迴歸任務通常為 1
    loss: str = 'mse'             # 此處不會用字串，而是會傳入自定義 loss 函數
    final_activation = None       # 迴歸任務不使用激活函數

# ---------------------------
# 修改後的 Self-Attention 模組
# Q 從中間層輸入取得，K 與 V 從原始資料取得
# ---------------------------
class SelfAttention(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = layers.Dense(dim)
        self.k_proj = layers.Dense(dim)
        self.v_proj = layers.Dense(dim)  # 新增 V 的投影
        self.softmax = layers.Softmax(axis=-1)
        self.out_proj = layers.Dense(dim)

    def call(self, query, key_value):
        # query 來源：中間層 (middle_output)
        # key_value 來源：原始資料 x
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        attention_scores = self.softmax(tf.matmul(Q, K, transpose_b=True))
        attention_output = tf.matmul(attention_scores, V)
        return self.out_proj(attention_output)

# ---------------------------
# 定義 SSMCausalConvBlock 模組（保持原有程式碼）
# ---------------------------
class SSMCausalConvBlock(layers.Layer):
    def __init__(self, modelargs: ModelArgs):
        super().__init__()
        self.args = modelargs
        # SSM 參數（隨機初始化，僅供示範）
        self.A_log = tf.Variable(tf.random.normal([modelargs.model_input_dims, modelargs.model_states]))
        self.D = tf.Variable(tf.ones([modelargs.model_input_dims]))
        self.x_projection = layers.Dense(modelargs.model_states * 2 + modelargs.model_input_dims, use_bias=False)
        self.delta_t_projection = layers.Dense(modelargs.model_input_dims, use_bias=True)
        # 因果捲積（使用 causal padding 保持時間順序）
        self.conv1 = layers.Conv1D(filters=modelargs.model_input_dims, kernel_size=3, padding='causal', activation='relu')
        # SSM 輸出層
        self.linear_filter = layers.Dense(modelargs.model_input_dims, activation='relu')

    def call(self, x, h_t_minus_1):
        A = -tf.exp(self.A_log)
        D = self.D
        x_dbl = self.x_projection(x)
        # 分割為 delta, B, C
        delta, B, C = tf.split(x_dbl, [self.args.model_input_dims, self.args.model_states, self.args.model_states], axis=-1)
        delta = tf.nn.softplus(self.delta_t_projection(delta))
        # 計算新的隱藏狀態 h_t
        h_t = self.selective_scan(h_t_minus_1, delta, A, B, C, D)
        ssm_output = self.linear_filter(h_t)
        causal_output = self.conv1(ssm_output)
        fused_output = ssm_output * causal_output
        return fused_output, h_t

    def selective_scan(self, h_t_minus_1, delta, A, B, C, D):
        dA = tf.einsum('bld,dn->bldn', delta, A)
        dB_u = tf.einsum('bld,bld,bln->bldn', delta, h_t_minus_1, B)
        dA_cumsum = tf.exp(tf.cumsum(dA, axis=1, exclusive=True))
        x = tf.einsum('bldn,bln->bld', dB_u * dA_cumsum, C)
        return x + x * D

# ---------------------------
# 定義 MambaBlock（使用 SSMCausalConvBlock）
# ---------------------------
class MambaBlock(layers.Layer):
    def __init__(self, modelargs: ModelArgs):
        super().__init__()
        self.conv1 = layers.Conv1D(
            filters=modelargs.model_input_dims, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        )
        self.ssm_block = SSMCausalConvBlock(modelargs)
        self.out_proj = layers.Dense(modelargs.model_input_dims)

    def call(self, x):
        x = self.conv1(x)
        h0 = tf.zeros_like(x)
        ssm_out, _ = self.ssm_block(x, h0)
        return self.out_proj(ssm_out)

# ---------------------------
# 定義完整模型結構 (Mamba Virtual Measurement)
# 輸出為列表：[mean, upper, lower]
# ---------------------------
def build_model(args: ModelArgs):
    input_layer = layers.Input(shape=(9, 9, 1), name='input')
    seq_length = 9 * 9
    x = layers.Reshape((seq_length, 1))(input_layer)
    x = layers.Dense(args.model_input_dims)(x)
    
    # Middle branch (mean)
    middle_output = MambaBlock(args)(x)
    
    # Upper branch：自注意力的 query 來自 middle_output，K 與 V 使用原始 x
    attention_upper = SelfAttention(args.model_input_dims)(middle_output, x)
    upper_input = layers.Add()([x, attention_upper])
    upper_output = MambaBlock(args)(upper_input)
    
    # Lower branch：同理，使用 middle_output 作為 query
    attention_lower = SelfAttention(args.model_input_dims)(middle_output, x)
    lower_input = layers.Add()([x, attention_lower])
    lower_output = MambaBlock(args)(lower_input)
    
    middle_flat = layers.Flatten()(middle_output)
    upper_flat = layers.Flatten()(upper_output)
    lower_flat = layers.Flatten()(lower_output)
    
    mean_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='mean')(middle_flat)
    upper_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='upper')(upper_flat)
    lower_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='lower')(lower_flat)
    
    model = Model(inputs=input_layer, outputs=[mean_pred, upper_pred, lower_pred], name='Mamba_Virtual_Measurement')
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=loss_mse, optimizer=adam)
    return model

# ---------------------------
# 自定義 Loss 函數（全部改為純 TensorFlow 運算，不使用 .numpy()）
# ---------------------------
def loss_mse(y_true, y_pred):
    # y_pred 為列表：[mean, upper, lower]
    theta = 1.0
    total_loss = 0.0
    loss_list = []
    # 分支 0：mean 直接用 MSE
    l0 = MSE(y_pred[0], y_true)
    loss_list.append(l0)
    total_loss += l0
    # 分支 1：upper
    raw = y_true - 65.0
    lambda_val = tf.where(raw < theta, 0.0, raw)
    lambda_base = tf.where(lambda_val > theta, 1.0, lambda_val)
    lambda_modified = tf.where(tf.equal(lambda_val, 0.0),
                               0.0,
                               tf.where(y_true >= 70.0, 1.0, tf.math.tanh(lambda_val)))
    total_lambda = lambda_base + lambda_modified
    l1 = loss_mse_lambda(y_pred[1], y_true, total_lambda)
    loss_list.append(l1)
    total_loss += l1
    # 分支 2：lower
    raw = 65.0 - y_true
    lambda_val = tf.where(raw < theta, 0.0, raw)
    lambda_base = tf.where(lambda_val > theta, 1.0, lambda_val)
    lambda_modified = tf.where(tf.equal(lambda_val, 0.0),
                               0.0,
                               tf.where(y_true <= 60.0, 1.0, tf.math.tanh(lambda_val)))
    total_lambda = lambda_base + lambda_modified
    l2 = loss_mse_lambda(y_pred[2], y_true, total_lambda)
    loss_list.append(l2)
    total_loss += l2
    return total_loss

def MSE(y_pred, y_true):  
    return tf.reduce_mean(tf.square(y_true - y_pred))

def loss_mse_lambda(outputs, targets, _lambda):
    # outputs, targets, _lambda 均形狀為 [batch, 1]
    sq_error = tf.square(outputs - targets)
    weighted_loss = _lambda * sq_error
    total_loss = tf.reduce_sum(weighted_loss)
    count = tf.reduce_sum(tf.cast(_lambda > 0, tf.float32))
    count = tf.maximum(count, 1.0)
    return total_loss / count

def loss_mse_lambda_2(outputs, outputs2, targets, _lambda):
    diff = tf.square(outputs - targets) - tf.square(outputs2 - targets)
    weighted_loss = _lambda * diff
    total_loss = tf.reduce_sum(weighted_loss)
    count = tf.reduce_sum(tf.cast(_lambda > 0, tf.float32))
    count = tf.maximum(count, 1.0)
    return total_loss / count

# ---------------------------
# 載入資料（假設 npy 檔案中資料形狀為 (num_samples, 9, 9)）
# ---------------------------
CP_data_train_2d_X = np.load('./cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_train_2d_Y = np.load('./cnn-2d_2020-09-09_11-45-24_y.npy')
CP_data_train_2d_X_tensor = tf.convert_to_tensor(CP_data_train_2d_X)
CP_data_train_2d_Y_tensor = tf.convert_to_tensor(CP_data_train_2d_Y)
train_X_2d = tf.expand_dims(CP_data_train_2d_X_tensor, axis=3)
train_Y_2d = CP_data_train_2d_Y_tensor
print("Training X shape:", train_X_2d.shape)
print("Training Y shape:", train_Y_2d.shape)

CP_data_valid_2d_X = np.load('./validation_data/cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_valid_2d_Y = np.load('./validation_data/cnn-2d_2020-09-09_11-45-24_y.npy')
CP_data_valid_2d_X_tensor = tf.convert_to_tensor(CP_data_valid_2d_X)
CP_data_valid_2d_Y_tensor = tf.convert_to_tensor(CP_data_valid_2d_Y)
valid_X_2d = tf.expand_dims(CP_data_valid_2d_X_tensor, axis=3)
valid_Y_2d = CP_data_valid_2d_Y_tensor
print("Validation X shape:", valid_X_2d.shape)
print("Validation Y shape:", valid_Y_2d.shape)

CP_data_test_2d_X=np.load('./testing_data/cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_test_2d_Y=np.load('./testing_data/cnn-2d_2020-09-09_11-45-24_y.npy')

CP_data_test_2d_X_tensor=tf.convert_to_tensor(CP_data_test_2d_X)
CP_data_test_2d_Y_tensor=tf.convert_to_tensor(CP_data_test_2d_Y)

test_X_2d = tf.expand_dims(CP_data_test_2d_X_tensor, axis=3)
test_Y_2d = CP_data_test_2d_Y_tensor 

#看train訓練完儲存的hdf5做更改名稱()
#checkpoint_path = "checkpoint/weights.05-796.18.hdf5"
checkpoint_path = "checkpoint/123.hdf5"
# ---------------------------
# 初始化模型參數並建立模型
# ---------------------------
args = ModelArgs(
    model_input_dims=128,
    model_states=32,
    num_layers=5,
    dropout_rate=0.2,
    num_classes=1,
    loss=loss_mse
)

model=build_model(args)
model.summary()

# ---------------------------
# 開始訓練
# ---------------------------
epochs = 20000
history = model.fit(train_X_2d, train_Y_2d, batch_size=100, epochs=epochs, validation_data=(valid_X_2d, valid_Y_2d), validation_freq=10)

# ---------------------------
#預測與評估（以下部分保持不變）
# ---------------------------
pred_y = model.predict(train_X_2d)
pred_valid_y = model.predict(valid_X_2d)
print("模型目標：", train_Y_2d)
print("valid模型目標：", valid_Y_2d)
pred_y_tensor = tf.convert_to_tensor(pred_y)
print("train模型預測：", pred_y_tensor)
pred_valid_y_tensor = tf.convert_to_tensor(pred_valid_y)
print("valid模型預測：", pred_valid_y_tensor)

def mape_np(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MAE_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


#直接抓訓練好的
model.load_weights(checkpoint_path)
pred_y = model.predict(test_X_2d)
pred_y_tensor=tf.convert_to_tensor(pred_y)

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape

def MAPE(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape

def MAE(y_true, y_pred):
    mae=np.mean(np.abs(y_true - y_pred))
    return mae

mean = 65.0
theta = 2.0
mean_range_upper = mean + theta
mean_range_lower = mean - theta
boundary_upper = 70.0
boundary_lower = 60.0

# ---------------------------
############################## Classifier->use pred_test_net_racnn[0] result as classifier ##############################
# ---------------------------

pred_test_net_cls = np.copy(pred_y[0]) 
print(pred_test_net_cls)

whole_mae_data_BN0 = np.round(MAE(test_Y_2d,pred_test_net_cls), 5)
#print("whole_mae_data_BN0 ：",whole_mae_data_BN0)

whole_mape_data_BN0 = np.round(MAPE(test_Y_2d,pred_test_net_cls), 5)
#print("whole_mape_data_BN0 ：",whole_mape_data_BN0)

for idx, data in enumerate(pred_test_net_cls):
    if(mean_range_lower <= data.item() <= mean_range_upper): # mean range
        pred_test_net_cls[idx] = 0
    elif(data.item() > mean_range_upper): # upper range
        pred_test_net_cls[idx] = 1
    elif(data.item() < mean_range_lower): # lower range
        pred_test_net_cls[idx] = 2
    #print(f"pred_test_net_cls[idx]: {pred_test_net_cls[idx]}")
    # print("*"*50)
#print(pred_test_net_cls)
pred_test_net_cls =tf.squeeze(pred_test_net_cls)
#print("pred_test_net_cls：",pred_test_net_cls)
pred_test_net_cls=pred_test_net_cls.numpy().astype('int64')
#print(type(pred_test_net_cls))

############################## RACNN+Classifier: 整合結果之陣列(均＋上＋下) ##############################
pred_test_net_combine_org = np.copy(pred_y[0]) 
for idx, cls_result in enumerate(pred_test_net_cls): # 整合 Classifier Predict Result
    #print(f"cls_result: {cls_result}, b0: {pred_y[0][idx]}, b1: {pred_y[1][idx]}, b2: {pred_y[2][idx]}")
    pred_test_net_combine_org[idx] = pred_y[cls_result][idx]
    #print(f"pred_test_net_combine[idx]: {pred_test_net_combine[idx]}")

#pred_y=tf.convert_to_tensor(pred_y)
pred_test_net_combine_org=tf.convert_to_tensor(pred_test_net_combine_org)
print("pred_test_net_combine：",pred_test_net_combine_org)

#print(type(pred_test_net_combine.numpy()))
whole_mape_data = np.round(MAPE(test_Y_2d,pred_test_net_combine_org.numpy()), 5)
print("whole_mape_data ：",whole_mape_data)

whole_MAE_data = np.round(MAE(test_Y_2d,pred_test_net_combine_org.numpy()), 5)
print("whole_MAE_data ：",whole_MAE_data)

a_mean=np.mean(pred_test_net_combine_org)


plt.figure(figsize=(10,12))
plt.subplot(4,1,1)
plt.title("HMamba-BN0")
plt.xlabel('Process Data(X)')
plt.ylabel('CoM')
plt.plot(test_Y_2d,marker=".",linestyle="-",color='blue',label='Ground Truch-CoM')
plt.plot(pred_y[0],marker="o",linestyle="-",color='orange',label='Model Predict-CoM')
plt.axhline(y=70,color='green',label='Boundary Upper')
plt.axhline(y=66,color='magenta',linestyle="--",label='Boundary Lower')
plt.axhline(y=64,color='blue',linestyle="--",label='Sensitivity Upper')
plt.axhline(y=60,color='red',label='Sensitivity Upper')
plt.legend(frameon=False,bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)


plt.subplot(4,1,2)
plt.title("HMamba-upper")
plt.xlabel('Process Data(X)')
plt.ylabel('CoM')
plt.plot(test_Y_2d,marker=".",linestyle="-",color='blue',label='Ground Truch-CoM')
plt.plot(pred_y[1],marker="o",linestyle="-",color='orange',label='Model Predict-CoM')
plt.axhline(y=70,color='green',label='Boundary Upper')
plt.axhline(y=66,color='magenta',linestyle="--",label='Boundary Lower')
plt.axhline(y=64,color='blue',linestyle="--",label='Sensitivity Upper')
plt.axhline(y=60,color='red',label='Sensitivity Upper')
plt.legend(frameon=False,bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)


plt.subplot(4,1,3)
plt.title("HMamba-lower")
plt.xlabel('Process Data(X)')
plt.ylabel('CoM')
plt.plot(test_Y_2d,marker=".",linestyle="-",color='blue',label='Ground Truch-CoM')
plt.plot(pred_y[2],marker="o",linestyle="-",color='orange',label='Model Predict-CoM')
plt.axhline(y=70,color='green',label='Boundary Upper')
plt.axhline(y=66,color='magenta',linestyle="--",label='Boundary Lower')
plt.axhline(y=64,color='blue',linestyle="--",label='Sensitivity Upper')
plt.axhline(y=60,color='red',label='Sensitivity Upper')
plt.legend(frameon=False,bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)

#只輸出HCNN圖
plt.subplot(1,1,1)
#plt.title("HMamba")
plt.xlabel('Processed PET Bottle Index(X)')
plt.ylabel('Virtual Metrology Value(mm)')
plt.plot(test_Y_2d,marker="*",linestyle=" ",color='blue',label='Ground Truch-CoM',markeredgecolor='black')
#plt.plot(pred_test_net_combine,marker="o",linestyle="-",color='palegreen',label='HCNN Model Predict-CoM',markeredgecolor='black')
plt.plot(pred_test_net_combine_org,marker="o",linestyle="-",color='orange',label='Mamba Model Predict-CoM',markeredgecolor='black')
plt.axhline(y=70,color='green',label='Boundary Upper')
plt.axhline(y=66,color='magenta',linestyle="--",label='Boundary Lower')
plt.axhline(y=64,color='blue',linestyle="--",label='Sensitivity Upper')
plt.axhline(y=60,color='red',label='Sensitivity Upper')
plt.legend(frameon=False,bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)
plt.show()
