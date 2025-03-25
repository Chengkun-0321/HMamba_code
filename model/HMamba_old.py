import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from dataclasses import dataclass

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
# 定義 Self-Attention 模組
# ---------------------------
class SelfAttention(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = layers.Dense(dim)
        self.k_proj = layers.Dense(dim)
        self.softmax = layers.Softmax(axis=-1)
        self.out_proj = layers.Dense(dim)

    def call(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        attention_scores = self.softmax(tf.matmul(Q, K, transpose_b=True))
        return self.out_proj(attention_scores)

# ---------------------------
# 定義 SSMCausalConvBlock 模組
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
    # Upper branch
    attention_upper = SelfAttention(args.model_input_dims)(x)
    upper_input = layers.Add()([x, attention_upper])
    upper_output = MambaBlock(args)(upper_input)
    # Lower branch
    attention_lower = SelfAttention(args.model_input_dims)(x)
    lower_input = layers.Add()([x, attention_lower])
    lower_output = MambaBlock(args)(lower_input)
    middle_flat = layers.Flatten()(middle_output)
    upper_flat = layers.Flatten()(upper_output)
    lower_flat = layers.Flatten()(lower_output)
    mean_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='mean')(middle_flat)
    upper_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='upper')(upper_flat)
    lower_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='lower')(lower_flat)
    model = Model(inputs=input_layer, outputs=[mean_pred, upper_pred, lower_pred], name='Mamba_Virtual_Measurement')
    adam=keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=loss_mse, optimizer=adam)
    #model.compile(loss=loss_mse, optimizer='adam', metrics=['mape'])
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
    #l1 += tf.clip_by_value(loss_mse_lambda_2(y_pred[1], y_pred[0], y_true, total_lambda), 0.0, 10000.0)
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
    #l2 += tf.clip_by_value(loss_mse_lambda_2(y_pred[2], y_pred[0], y_true, total_lambda), 0.0, 10000.0)
    loss_list.append(l2)
    total_loss += l2
    #tf.print(loss_list)
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


#checkpoint_path = "checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
#callback=[keras.callbacks.CSVLogger('./result_data/log.csv', separator=',', append=False)]
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True,period=2)
#tf_callback = tf.keras.callbacks.TensorBoard(log_dir="log")

model = build_model(args)
model.summary()
# ---------------------------
# 開始訓練
# ---------------------------
epochs=20000
history = model.fit(train_X_2d,train_Y_2d,batch_size=100,epochs=epochs,validation_data=(valid_X_2d, valid_Y_2d),validation_freq=10)
#history = model.fit(train_X_2d,train_Y_2d,batch_size=1000,epochs=epochs,validation_data=(valid_X_2d, valid_Y_2d),validation_freq=2,callbacks=[tf_callback,callback,cp_callback])
# ---------------------------
# 預測與評估
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

train_mape_data_b0 = np.round(mape_np(train_Y_2d, pred_y[0]), 5)
train_mape_data_b1 = np.round(mape_np(train_Y_2d, pred_y[1]), 5)
train_mape_data_b2 = np.round(mape_np(train_Y_2d, pred_y[2]), 5)
valid_mape_data_b0 = np.round(mape_np(valid_Y_2d, pred_valid_y[0]), 5)
valid_mape_data_b1 = np.round(mape_np(valid_Y_2d, pred_valid_y[1]), 5)
valid_mape_data_b2 = np.round(mape_np(valid_Y_2d, pred_valid_y[2]), 5)

train_mae_data_b0 = np.round(MAE_np(train_Y_2d, pred_y[0]), 5)
train_mae_data_b1 = np.round(MAE_np(train_Y_2d, pred_y[1]), 5)
train_mae_data_b2 = np.round(MAE_np(train_Y_2d, pred_y[2]), 5)
valid_mae_data_b0 = np.round(MAE_np(valid_Y_2d, pred_valid_y[0]), 5)
valid_mae_data_b1 = np.round(MAE_np(valid_Y_2d, pred_valid_y[1]), 5)
valid_mae_data_b2 = np.round(MAE_np(valid_Y_2d, pred_valid_y[2]), 5)

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

mean_range_upper = 66.0
mean_range_lower = 64.0
pred_train_net_combine = np.copy(pred_y_tensor[0])
pred_valid_net_combine = np.copy(pred_valid_y_tensor[0])

upper_train_gt, upper_train_pred, lower_train_gt, lower_train_pred = [], [], [], []
upper_valid_gt, upper_valid_pred, lower_valid_gt, lower_valid_pred = [], [], [], []

for idx, y in enumerate(train_Y_2d):
    # 注意：此處使用 .numpy() 需在 eager 模式下
    if (mean_range_lower <= y.numpy() <= mean_range_upper):
        pred_train_net_combine[idx] = pred_y[0][idx]
    elif (y.numpy() > mean_range_upper):
        pred_train_net_combine[idx] = pred_y[1][idx]
        upper_train_gt.append(y.numpy())
        upper_train_pred.append(pred_y[1][idx])
    elif (y.numpy() < mean_range_lower):
        pred_train_net_combine[idx] = pred_y[2][idx]
        lower_train_gt.append(y.numpy())
        lower_train_pred.append(pred_y[2][idx])

for idx, y in enumerate(valid_Y_2d):
    if (mean_range_lower <= y.numpy() <= mean_range_upper):
        pred_valid_net_combine[idx] = pred_valid_y[0][idx]
    elif (y.numpy() > mean_range_upper):
        pred_valid_net_combine[idx] = pred_valid_y[1][idx]
        upper_valid_gt.append(y.numpy())
        upper_valid_pred.append(pred_valid_y[1][idx])
    elif (y.numpy() < mean_range_lower):
        pred_valid_net_combine[idx] = pred_valid_y[2][idx]
        lower_valid_gt.append(y.numpy())
        lower_valid_pred.append(pred_valid_y[2][idx])

upper_train_pred = np.array(upper_train_pred)
upper_train_gt = np.array(upper_train_gt)
lower_train_pred = np.array(lower_train_pred)
lower_train_gt = np.array(lower_train_gt)
upper_valid_pred = np.array(upper_valid_pred)
upper_valid_gt = np.array(upper_valid_gt)
lower_valid_pred = np.array(lower_valid_pred)
lower_valid_gt = np.array(lower_valid_gt)

upper_train_mape_data = np.round(mape_np(upper_train_pred, upper_train_gt), 5)
lower_train_mape_data = np.round(mape_np(lower_train_pred, lower_train_gt), 5)
upper_valid_mape_data = np.round(mape_np(upper_valid_pred, upper_valid_gt), 5)
lower_valid_mape_data = np.round(mape_np(lower_valid_pred, lower_valid_gt), 5)

upper_train_mae_data = np.round(MAE_np(upper_train_pred, upper_train_gt), 5)
lower_train_mae_data = np.round(MAE_np(lower_train_pred, lower_train_gt), 5)
upper_valid_mae_data = np.round(MAE_np(upper_valid_pred, upper_valid_gt), 5)
lower_valid_mae_data = np.round(MAE_np(lower_valid_pred, lower_valid_gt), 5)

print("\n===================上下界MAPE======================")
print("upper_train_mape_data:", upper_train_mape_data)
print("lower_train_mape_data:", lower_train_mape_data)
print("upper_valid_mape_data:", upper_valid_mape_data)
print("lower_valid_mape_data:", lower_valid_mape_data)
print("\n===================上下界MAE======================")
print("upper_train_mae_data:", upper_train_mae_data)
print("lower_train_mae_data:", lower_train_mae_data)
print("upper_valid_mae_data:", upper_valid_mae_data)
print("lower_valid_mae_data:", lower_valid_mae_data)

integr_train_mape_data_whole = np.round(mape_np(pred_train_net_combine, train_Y_2d), 5)
integr_valid_mape_data_whole = np.round(mape_np(pred_valid_net_combine, valid_Y_2d), 5)
integr_train_mae_data_whole = np.round(MAE_np(pred_train_net_combine, train_Y_2d), 5)
integr_valid_mae_data_whole = np.round(MAE_np(pred_valid_net_combine, valid_Y_2d), 5)
print("\n===================整合MAPE======================")
print("integr_train_mape_data_whole:", integr_train_mape_data_whole)
print("integr_valid_mape_data_whole:", integr_valid_mape_data_whole)
print("\n===================整合MAE======================")
print("integr_train_mae_data_whole:", integr_train_mae_data_whole)
print("integr_valid_mae_data_whole:", integr_valid_mae_data_whole)