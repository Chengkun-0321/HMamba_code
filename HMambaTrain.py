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
import argparse
#import util2 as u   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,LeakyReLU,MaxPool2D,Dense,Flatten

tf.config.run_functions_eagerly(True)
# ---------------------------
# 定義模型參數
# ---------------------------
@dataclass
class ModelArgs:
    model_input_dims: int = 128   
    model_states: int = 32        
    num_layers: int = 5           
    dropout_rate: float = 0.2     
    num_classes: int = 1          
    loss: str = 'mse'             
    final_activation = None    
    learning_rate: float = 0.001  

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
    '''
    def selective_scan(self, h_t_minus_1, delta, A, B, C, D):
        dA = tf.einsum('bld,dn->bldn', delta, A)
        dB_u = tf.einsum('bld,bld,bln->bldn', delta, h_t_minus_1, B)
        dA_cumsum = tf.exp(tf.cumsum(dA, axis=1, exclusive=True))
        x = tf.einsum('bldn,bln->bld', dB_u * dA_cumsum, C)
        return x + x * D
    '''
    def selective_scan(self, h_t_minus_1, delta, A, B, C, D):
        # 利用 einsum 計算 dA，shape 為 (batch, seq_len, model_input_dims, model_states)
        dA = tf.einsum('bld,dn->bldn', delta, A)
        # 對 dA 進行 clipping 防止過大值，範圍可依需求調整
        dA = tf.clip_by_value(dA, -5.0, 5.0)
        # 利用 einsum 計算 dB_u
        dB_u = tf.einsum('bld,bld,bln->bldn', delta, h_t_minus_1, B)
        # 對 dA 的累積和做 clipping，防止 exp 輸入過大
        cumsum_dA = tf.cumsum(dA, axis=1, exclusive=True)
        cumsum_dA = tf.clip_by_value(cumsum_dA, -10.0, 10.0)
        dA_cumsum = tf.exp(cumsum_dA)
        # 計算狀態更新
        x_state = tf.einsum('bldn,bln->bld', dB_u * dA_cumsum, C)
        h_t_out = x_state + x_state * D
        # 最後再對狀態做一次 clipping
        return tf.clip_by_value(h_t_out, -100.0, 100.0)

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
        x_conv = self.conv1(x)
        # 使用 conv1 輸出作為初始隱藏狀態（避免全零導致數值計算不更新）
        h0 = x_conv
        ssm_out, _ = self.ssm_block(x_conv, h0)
        return self.out_proj(ssm_out)
        '''
    def call(self, x):
        x = self.conv1(x)
        h0 = tf.zeros_like(x)
        ssm_out, _ = self.ssm_block(x, h0)
        return self.out_proj(ssm_out)
    '''

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

    output = layers.Concatenate(axis=-1)([mean_pred, upper_pred, lower_pred])
    #model = Model(inputs=input_layer, outputs=[mean_pred, upper_pred, lower_pred], name='Mamba_Virtual_Measurement')
    # 利用從 ModelArgs 帶入的 learning_rate 建立 Adam 優化器
    adam = keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
    # 使用自訂 loss 函數 loss_mse 編譯模型
    model = Model(inputs=input_layer, outputs=output, name='Mamba_Virtual_Measurement')
    model.compile(loss=loss_mse, optimizer=adam)
    return model

# ---------------------------
# 自定義 Loss 函數
# ---------------------------

def loss_mse(y_true, y_pred):
        output = tf.transpose(y_pred)
        y_pred = tf.expand_dims(output,2)
        theta=1
        loss_mse = 0
        _loss_mse = [] 
        for i in range(len(y_pred)):
            if(i == 0): # 均
                l = 0 
                l = MSE(y_pred[0], y_true)
                _loss_mse.append(l)
                loss_mse += l
            elif(i == 1): # 上
                l = 0 
                lambda_ = y_true - 65
                lambda_=tf.where(lambda_ < theta, 0, lambda_)
                lambda_base=np.copy(lambda_)
                lambda_base=tf.where(lambda_ > theta, 1, lambda_)
                for idx, data in enumerate(lambda_):                    
                    if(data.numpy() == 0): continue
                    elif(y_true.numpy()[idx]>=70.0): # 位於上界之上 
                        lambda_=tf.Variable(lambda_)
                        lambda_=lambda_.numpy()
                        lambda_[idx] = 1
                    else:
                        data=data.numpy()
                        data2=float(data)
                        a= tf.math.tanh(data2)
                        lambda_=tf.Variable(lambda_)
                        lambda_=lambda_.numpy()
                        lambda_[idx]=a
                total_lambda = lambda_base + lambda_
                l += loss_mse_lambda(y_pred[1], y_true, total_lambda)
                l += tf.clip_by_value((loss_mse_lambda_2(y_pred[1], y_pred[0], y_true, total_lambda)), clip_value_min=0, clip_value_max=10000)
                _loss_mse.append(l)
                loss_mse += l
            elif(i == 2): # 下
                l = 0 
                lambda_ = 65 - y_true 
                lambda_=tf.where(lambda_ < theta, 0, lambda_)
                lambda_base=np.copy(lambda_)
                lambda_base=tf.where(lambda_ > theta, 1, lambda_)
                for idx, data in enumerate(lambda_):                 
                    if(data.numpy() == 0): continue
                    elif(y_true.numpy()[idx]<=60.0): 
                        lambda_=tf.Variable(lambda_)
                        lambda_=lambda_.numpy()
                        lambda_[idx] = 1
                    else:
                        data=data.numpy()
                        data2=float(data)
                        a= tf.math.tanh(data2)
                        lambda_=tf.Variable(lambda_)
                        lambda_=lambda_.numpy()
                        lambda_[idx]=a          
                total_lambda = lambda_base + lambda_
                l += loss_mse_lambda(y_pred[2], y_true, total_lambda)
                l += tf.clip_by_value((loss_mse_lambda_2(y_pred[2], y_pred[0], y_true, total_lambda)), clip_value_min=0, clip_value_max=10000)
                _loss_mse.append(l)
                loss_mse += l
        return  loss_mse     

'''
def loss_mse(y_true, y_pred):
    """
    y_pred 為 list：[mean_pred, upper_pred, lower_pred]
    0：均值分支
    1：上界分支
    2：下界分支
    """
    # 將 list 轉為 tensor，形狀變為 (3, batch, 1)
    y_pred = tf.stack(y_pred, axis=0)
    theta = 1.0
    loss_total = 0.0
    _loss_mse = []
    # 分支 0：mean 分支
    l = MSE(y_pred[0], y_true)
    _loss_mse.append(l)
    loss_total += l
    # 分支 1：upper 分支
    l = 0.0
    lambda_ = y_true - 65.0
    lambda_ = tf.where(lambda_ < theta, 0.0, lambda_)
    # 先複製一份作為 base
    lambda_base = np.copy(lambda_.numpy())
    lambda_base = tf.where(lambda_ > theta, 1.0, lambda_)
    # 轉換為 numpy 陣列以便逐筆處理（注意：在 eager mode 下才可使用 .numpy()）
    lambda_np = lambda_.numpy()
    for idx, data in enumerate(lambda_np):
        if data[0] == 0:
            continue
        elif y_true.numpy()[idx,0] >= 70.0:  # 位於上界之上
            lambda_np[idx,0] = 1.0
        else:
            a = tf.math.tanh(data[0])
            lambda_np[idx,0] = a.numpy()
    total_lambda = lambda_base + lambda_np
    l += loss_mse_lambda(y_pred[1], y_true, total_lambda)
    _loss_mse.append(l)
    loss_total += l
    # 分支 2：lower 分支
    l = 0.0
    lambda_ = 65.0 - y_true
    lambda_ = tf.where(lambda_ < theta, 0.0, lambda_)
    lambda_base = np.copy(lambda_.numpy())
    lambda_base = tf.where(lambda_ > theta, 1.0, lambda_)
    lambda_np = lambda_.numpy()
    for idx, data in enumerate(lambda_np):
        if data[0] == 0:
            continue
        elif y_true.numpy()[idx,0] <= 60.0:
            lambda_np[idx,0] = 1.0
        else:
            a = tf.math.tanh(data[0])
            lambda_np[idx,0] = a.numpy()
    total_lambda = lambda_base + lambda_np
    l += loss_mse_lambda(y_pred[2], y_true, total_lambda)
    _loss_mse.append(l)
    loss_total += l
    return loss_total
'''
def MSE(y_pred,y_true):  
    Square_Error = tf.square(y_true-y_pred)
    mse = tf.reduce_mean(Square_Error) 
    return mse

def MAE(y_pred,y_true):
    mae=np.mean(np.abs(y_true - y_pred))
    return mae 

def loss_mse_lambda(outputs, targets, _lambda): # 帶有lambda的MSE Loss計算Function
        total_loss = 0
        for idx,_ in enumerate(outputs):
            total_loss += _lambda[idx]*(outputs[idx]-targets[idx])**2
        lambda_elements_larger_than_0 = 0
        for lambda_val in _lambda:
            if(lambda_val[0] > 0): lambda_elements_larger_than_0+=1
        if(lambda_elements_larger_than_0 == 0): lambda_elements_larger_than_0 = 1
        X = tf.cast(lambda_elements_larger_than_0,dtype=float)
        a=total_loss/X
        return a[0] 

def loss_mse_lambda_2(outputs, outputs2 ,targets, _lambda): # 帶有lambda的MSE Loss計算Function
        total_loss = 0
        for idx,_ in enumerate(outputs):
            total_mse=(outputs[idx]-targets[idx])**2 -(outputs2[idx]-targets[idx])**2 
            total_loss +=  _lambda[idx]*total_mse
        lambda_elements_larger_than_0 = 0
        for lambda_val in _lambda:
            if(lambda_val[0] > 0): lambda_elements_larger_than_0+=1
        if(lambda_elements_larger_than_0 == 0): lambda_elements_larger_than_0 = 1
        X = tf.cast(lambda_elements_larger_than_0,dtype=float)
        a=total_loss/X
        return a[0] 

# ---------------------------
# 其他輔助函數 (MAPE 與 MAE)
# ---------------------------
def mape_np(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MAE_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# ---------------------------
# 主程式，利用 argparse 讀取參數並執行訓練與評估
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Mamba Virtual Measurement Training Script"
    )
    parser.add_argument("--train_x", type=str, required=True,
                        help="Path to training X numpy file (e.g. './cnn-2d_2020-09-09_11-45-24_x.npy')")
    parser.add_argument("--train_y", type=str, required=True,
                        help="Path to training Y numpy file (e.g. './cnn-2d_2020-09-09_11-45-24_y.npy')")
    parser.add_argument("--valid_x", type=str, required=True,
                        help="Path to validation X numpy file")
    parser.add_argument("--valid_y", type=str, required=True,
                        help="Path to validation Y numpy file")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Training batch size (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--validation_freq", type=int, default=5,
                        help="Run validation every n epochs (default: 5)")
    args_cli = parser.parse_args()

    # 建立 checkpoint 資料夾（若不存在的話）
    os.makedirs("./checkpoint", exist_ok=True)

    # ---------------------------
    # 載入訓練資料
    # ---------------------------
    CP_data_train_2d_X = np.load(args_cli.train_x)
    CP_data_train_2d_Y = np.load(args_cli.train_y)
    CP_data_train_2d_X_tensor = tf.convert_to_tensor(CP_data_train_2d_X, dtype=tf.float32)
    CP_data_train_2d_Y_tensor = tf.convert_to_tensor(CP_data_train_2d_Y, dtype=tf.float32)
    train_X_2d = tf.expand_dims(CP_data_train_2d_X_tensor, axis=3)
    train_Y_2d = CP_data_train_2d_Y_tensor
    

    # ---------------------------
    # 載入驗證資料
    # ---------------------------
    CP_data_valid_2d_X = np.load(args_cli.valid_x)
    CP_data_valid_2d_Y = np.load(args_cli.valid_y)
    CP_data_valid_2d_X_tensor = tf.convert_to_tensor(CP_data_valid_2d_X, dtype=tf.float32)
    CP_data_valid_2d_Y_tensor = tf.convert_to_tensor(CP_data_valid_2d_Y, dtype=tf.float32)
    valid_X_2d = tf.expand_dims(CP_data_valid_2d_X_tensor, axis=3)
    valid_Y_2d = CP_data_valid_2d_Y_tensor
    
    print('---------- Train/Test Shape ----------')
    print("Train X shape:", train_X_2d.shape)
    print("Train Y shape:", train_Y_2d.shape)
    print("Validation X shape:", valid_X_2d.shape)
    print("Validation Y shape:", valid_Y_2d.shape)
    print('------------------------------------\n')

    # ---------------------------
    # 初始化模型參數並建立模型
    # ---------------------------
    model_args = ModelArgs(
        model_input_dims=128,
        model_states=32,
        num_layers=5,
        dropout_rate=0.2,
        num_classes=1,
        loss=loss_mse,  # 使用自訂 loss 函數
        learning_rate=args_cli.lr  # 從指令列取得學習率
    )
    model = build_model(model_args)
    model.summary()

    # 設定 checkpoint callback ./checkpoint
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='./checkpoint/weights_epoch_{epoch:02d}.h5')

    # ---------------------------
    # 開始訓練
    # ---------------------------
    history = model.fit(train_X_2d, train_Y_2d, 
                        batch_size=args_cli.batch_size, 
                        epochs=args_cli.epochs,
                        validation_data=(valid_X_2d, valid_Y_2d),
                        validation_freq=args_cli.validation_freq,
                        callbacks=[checkpoint_callback]
                        )
    #callbacks=[checkpoint_callback]
    # ---------------------------
    # 預測與評估
    # ---------------------------
    pred_y = model.predict(train_X_2d)  # 預測結果 shape 為 (num_samples, 3)
    pred_valid_y = model.predict(valid_X_2d)
    print("模型目標：", train_Y_2d)
    pred_y_tensor = tf.convert_to_tensor(pred_y)
    print("Train 模型預測：", pred_y_tensor)

    # 計算各分支的 MAPE 與 MAE（注意：以 pred_y[:,0] 代表 mean 分支）
    train_mape_data_b0 = np.round(mape_np(train_Y_2d.numpy(), pred_y[:, 0]), 5)
    train_mape_data_b1 = np.round(mape_np(train_Y_2d.numpy(), pred_y[:, 1]), 5)
    train_mape_data_b2 = np.round(mape_np(train_Y_2d.numpy(), pred_y[:, 2]), 5)
    valid_mape_data_b0 = np.round(mape_np(valid_Y_2d.numpy(), pred_valid_y[:, 0]), 5)
    valid_mape_data_b1 = np.round(mape_np(valid_Y_2d.numpy(), pred_valid_y[:, 1]), 5)
    valid_mape_data_b2 = np.round(mape_np(valid_Y_2d.numpy(), pred_valid_y[:, 2]), 5)

    train_mae_data_b0 = np.round(MAE_np(train_Y_2d.numpy(), pred_y[:, 0]), 5)
    train_mae_data_b1 = np.round(MAE_np(train_Y_2d.numpy(), pred_y[:, 1]), 5)
    train_mae_data_b2 = np.round(MAE_np(train_Y_2d.numpy(), pred_y[:, 2]), 5)
    valid_mae_data_b0 = np.round(MAE_np(valid_Y_2d.numpy(), pred_valid_y[:, 0]), 5)
    valid_mae_data_b1 = np.round(MAE_np(valid_Y_2d.numpy(), pred_valid_y[:, 1]), 5)
    valid_mae_data_b2 = np.round(MAE_np(valid_Y_2d.numpy(), pred_valid_y[:, 2]), 5)

    print("====================中上下網路MAPE======================")
    print("train_mape_data_b0 ：", train_mape_data_b0)
    print("train_mape_data_b1 ：", train_mape_data_b1)
    print("train_mape_data_b2 ：", train_mape_data_b2)
    print("valid_mape_data_b0 ：", valid_mape_data_b0)
    print("valid_mape_data_b1 ：", valid_mape_data_b1)
    print("valid_mape_data_b2 ：", valid_mape_data_b2)

    print("\n====================中上下網路MAE======================")
    print("train_mae_data_b0 ：", train_mae_data_b0)
    print("train_mae_data_b1 ：", train_mae_data_b1)
    print("train_mae_data_b2 ：", train_mae_data_b2)
    print("valid_mae_data_b0 ：", valid_mae_data_b0)
    print("valid_mae_data_b1 ：", valid_mae_data_b1)
    print("valid_mae_data_b2 ：", valid_mae_data_b2)

    # ---------------------------
    # 整合預測（根據中位範圍判斷使用哪個分支預測）
    # ---------------------------
    mean_range_upper = 66.0
    mean_range_lower = 64.0
    # 預設以 mean 分支預測
    pred_train_net_combine = np.copy(pred_y[:, 0])

    upper_train_gt, upper_train_pred, lower_train_gt, lower_train_pred = [], [], [], []

    train_Y_2d_np = train_Y_2d.numpy()
    for idx, y in enumerate(train_Y_2d_np):
        y_val = y[0]  # 假設 y 的 shape 為 (1,)
        if mean_range_lower <= y_val <= mean_range_upper:
            pred_train_net_combine[idx] = pred_y[idx, 0]
        elif y_val > mean_range_upper:
            pred_train_net_combine[idx] = pred_y[idx, 1]
            upper_train_gt.append(y_val)
            upper_train_pred.append(pred_y[idx, 1])
        elif y_val < mean_range_lower:
            pred_train_net_combine[idx] = pred_y[idx, 2]
            lower_train_gt.append(y_val)
            lower_train_pred.append(pred_y[idx, 2])

    upper_train_pred = np.array(upper_train_pred)
    upper_train_gt = np.array(upper_train_gt)
    lower_train_pred = np.array(lower_train_pred)
    lower_train_gt = np.array(lower_train_gt)

    upper_train_mape_data = np.round(mape_np(upper_train_gt, upper_train_pred), 5) if len(upper_train_gt) > 0 else None
    lower_train_mape_data = np.round(mape_np(lower_train_gt, lower_train_pred), 5) if len(lower_train_gt) > 0 else None

    upper_train_mae_data = np.round(MAE_np(upper_train_gt, upper_train_pred), 5) if len(upper_train_gt) > 0 else None
    lower_train_mae_data = np.round(MAE_np(lower_train_gt, lower_train_pred), 5) if len(lower_train_gt) > 0 else None

    print("\n===================上下界MAPE======================")
    print("upper_train_mape_data:", upper_train_mape_data)
    print("lower_train_mape_data:", lower_train_mape_data)
    print("\n===================上下界MAE======================")
    print("upper_train_mae_data:", upper_train_mae_data)
    print("lower_train_mae_data:", lower_train_mae_data)

    integr_train_mape_data_whole = np.round(mape_np(train_Y_2d_np, pred_train_net_combine), 5)
    integr_train_mae_data_whole = np.round(MAE_np(train_Y_2d_np, pred_train_net_combine), 5)
    print("\n===================整合MAPE======================")
    print("integr_train_mape_data_whole:", integr_train_mape_data_whole)
    print("\n===================整合MAE======================")
    print("integr_train_mae_data_whole:", integr_train_mae_data_whole)

if __name__ == '__main__':
    main()
