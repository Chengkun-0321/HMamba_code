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

# 強制以 eager mode 執行（loss 中使用 .numpy()）
tf.config.run_functions_eagerly(True)

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
    loss: str = 'mse'             # 此處不會用字串，而是會傳入自訂 loss 函數
    final_activation = None       # 迴歸任務不使用激活函數

# ---------------------------
# Self-Attention 模組 (Q 來自中間層，K 與 V 來自原始資料)
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
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        attention_scores = self.softmax(tf.matmul(Q, K, transpose_b=True))
        attention_output = tf.matmul(attention_scores, V)
        return self.out_proj(attention_output)

# ---------------------------
# SSMCausalConvBlock 模組（保持原有程式碼）
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
# MambaBlock（使用 SSMCausalConvBlock）
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
# 輸出為一個張量，內含 [mean_pred, upper_pred, lower_pred]（shape: (batch, 3)）
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
    
    mean_pred = layers.Dense(args.num_classes, activation=args.final_activation)(middle_flat)
    upper_pred = layers.Dense(args.num_classes, activation=args.final_activation)(upper_flat)
    lower_pred = layers.Dense(args.num_classes, activation=args.final_activation)(lower_flat)
    
    # 將三個預測結果串接成一個張量 (shape: (batch, 3))
    combined_output = layers.Concatenate(axis=-1)([mean_pred, upper_pred, lower_pred])
    
    model = Model(inputs=input_layer, outputs=combined_output, name='Mamba_Virtual_Measurement')
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    # 編譯模型時只會顯示單一 loss 值
    model.compile(loss=loss_mse, optimizer=adam)
    return model

# ---------------------------
# 自訂 Loss 函數：根據 mean 分支的 loss 動態調整 theta（介於 0.1 與 2.0 之間）
# ---------------------------
def loss_mse(y_true, y_pred):
    """
    y_pred 為 shape (batch, 3)，依序分別代表 mean_pred, upper_pred, lower_pred
    """
    # 拆分預測結果
    mean_pred  = y_pred[:, 0:1]
    upper_pred = y_pred[:, 1:2]
    lower_pred = y_pred[:, 2:3]
    # 計算 mean 分支的 MSE loss
    l_mean = MSE(mean_pred, y_true)
    # 根據 l_mean 動態調整 theta，並限制在 [0.1, 2.0] 範圍內
    constant = 50.0  # 此值可根據您的資料調整
    theta = tf.clip_by_value(l_mean / constant, 0.1, 2.0)
    loss_total = 0.0
    # 分支 0：mean 分支 loss
    loss_total += l_mean

    # 分支 1：upper 分支 loss
    l_upper = 0.0
    lambda_upper = y_true - 65.0
    lambda_upper = tf.where(lambda_upper < theta, 0.0, lambda_upper)
    # 複製一份作為基底（以 numpy 處理逐筆資料）
    lambda_base_upper = np.copy(lambda_upper.numpy())
    lambda_base_upper = tf.where(lambda_upper > theta, 1.0, lambda_upper)
    lambda_np_upper = lambda_upper.numpy()
    for idx, data in enumerate(lambda_np_upper):
        if data[0] == 0:
            continue
        elif y_true.numpy()[idx, 0] >= 70.0:
            lambda_np_upper[idx, 0] = 1.0
        else:
            a = tf.math.tanh(data[0])
            lambda_np_upper[idx, 0] = a.numpy()
    total_lambda_upper = lambda_base_upper + lambda_np_upper
    l_upper += loss_mse_lambda(upper_pred, y_true, total_lambda_upper)
    l_upper += tf.clip_by_value(loss_mse_lambda_2(upper_pred, mean_pred, y_true, total_lambda_upper),
                                 clip_value_min=0.0, clip_value_max=10000.0)
    loss_total += l_upper

    # 分支 2：lower 分支 loss
    l_lower = 0.0
    lambda_lower = 65.0 - y_true
    lambda_lower = tf.where(lambda_lower < theta, 0.0, lambda_lower)
    lambda_base_lower = np.copy(lambda_lower.numpy())
    lambda_base_lower = tf.where(lambda_lower > theta, 1.0, lambda_lower)
    lambda_np_lower = lambda_lower.numpy()
    for idx, data in enumerate(lambda_np_lower):
        if data[0] == 0:
            continue
        elif y_true.numpy()[idx, 0] <= 60.0:
            lambda_np_lower[idx, 0] = 1.0
        else:
            a = tf.math.tanh(data[0])
            lambda_np_lower[idx, 0] = a.numpy()
    total_lambda_lower = lambda_base_lower + lambda_np_lower
    l_lower += loss_mse_lambda(lower_pred, y_true, total_lambda_lower)
    loss_total += l_lower
    return loss_total

def MSE(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def loss_mse_lambda(outputs, targets, _lambda):
    total_loss = 0
    outputs_np = outputs.numpy()
    targets_np = targets.numpy()
    for idx in range(outputs_np.shape[0]):
        total_loss += _lambda[idx, 0] * (outputs_np[idx, 0] - targets_np[idx, 0])**2
    lambda_elements = sum(1 for lam in _lambda if lam[0] > 0)
    if lambda_elements == 0:
        lambda_elements = 1
    a = total_loss / float(lambda_elements)
    return tf.convert_to_tensor(a, dtype=tf.float32)

def loss_mse_lambda_2(outputs, outputs2, targets, _lambda):
    total_loss = 0
    outputs_np = outputs.numpy()
    outputs2_np = outputs2.numpy()
    targets_np = targets.numpy()
    for idx in range(outputs_np.shape[0]):
        diff = (outputs_np[idx, 0] - targets_np[idx, 0])**2 - (outputs2_np[idx, 0] - targets_np[idx, 0])**2
        total_loss += _lambda[idx, 0] * diff
    lambda_elements = sum(1 for lam in _lambda if lam[0] > 0)
    if lambda_elements == 0:
        lambda_elements = 1
    a = total_loss / float(lambda_elements)
    return tf.convert_to_tensor(a, dtype=tf.float32)

# ---------------------------
# 載入資料（假設 npy 檔案中資料形狀為 (num_samples, 9, 9)）
# ---------------------------
CP_data_train_2d_X = np.load('./cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_train_2d_Y = np.load('./cnn-2d_2020-09-09_11-45-24_y.npy')
CP_data_train_2d_X_tensor = tf.convert_to_tensor(CP_data_train_2d_X, dtype=tf.float32)
CP_data_train_2d_Y_tensor = tf.convert_to_tensor(CP_data_train_2d_Y, dtype=tf.float32)
train_X_2d = tf.expand_dims(CP_data_train_2d_X_tensor, axis=3)
train_Y_2d = CP_data_train_2d_Y_tensor
print("Training X shape:", train_X_2d.shape)
print("Training Y shape:", train_Y_2d.shape)

CP_data_valid_2d_X = np.load('./validation_data/cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_valid_2d_Y = np.load('./validation_data/cnn-2d_2020-09-09_11-45-24_y.npy')
CP_data_valid_2d_X_tensor = tf.convert_to_tensor(CP_data_valid_2d_X, dtype=tf.float32)
CP_data_valid_2d_Y_tensor = tf.convert_to_tensor(CP_data_valid_2d_Y, dtype=tf.float32)
valid_X_2d = tf.expand_dims(CP_data_valid_2d_X_tensor, axis=3)
valid_Y_2d = CP_data_valid_2d_Y_tensor
print("Validation X shape:", valid_X_2d.shape)
print("Validation Y shape:", valid_Y_2d.shape)

# ---------------------------
# 初始化模型並訓練
# ---------------------------
args = ModelArgs(
    model_input_dims=128,
    model_states=32,
    num_layers=5,
    dropout_rate=0.2,
    num_classes=1,
    loss=loss_mse
)

model = build_model(args)
model.summary()

# 訓練 (請根據資料量與需求調整 epochs 與 batch_size)
epochs = 30000
history = model.fit(train_X_2d, train_Y_2d, batch_size=1000, epochs=epochs,
                    validation_data=(valid_X_2d, valid_Y_2d), validation_freq=10)

# ---------------------------
# 預測與評估
# ---------------------------
pred_y = model.predict(train_X_2d)
pred_valid_y = model.predict(valid_X_2d)
print("Training Targets:", train_Y_2d)
print("Validation Targets:", valid_Y_2d)
pred_y_tensor = tf.convert_to_tensor(pred_y)
print("Training Predictions:", pred_y_tensor)
pred_valid_y_tensor = tf.convert_to_tensor(pred_valid_y)
print("Validation Predictions:", pred_valid_y_tensor)

def mape_np(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MAE_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# 因為模型輸出為一個張量 (batch, 3)，因此拆分後計算各分支指標
mean_pred  = pred_y[:, 0:1]
upper_pred = pred_y[:, 1:2]
lower_pred = pred_y[:, 2:3]
mean_pred_valid  = pred_valid_y[:, 0:1]
upper_pred_valid = pred_valid_y[:, 1:2]
lower_pred_valid = pred_valid_y[:, 2:3]

train_mape_data_b0 = np.round(mape_np(train_Y_2d, mean_pred), 5)
train_mape_data_b1 = np.round(mape_np(train_Y_2d, upper_pred), 5)
train_mape_data_b2 = np.round(mape_np(train_Y_2d, lower_pred), 5)
valid_mape_data_b0 = np.round(mape_np(valid_Y_2d, mean_pred_valid), 5)
valid_mape_data_b1 = np.round(mape_np(valid_Y_2d, upper_pred_valid), 5)
valid_mape_data_b2 = np.round(mape_np(valid_Y_2d, lower_pred_valid), 5)

train_mae_data_b0 = np.round(MAE_np(train_Y_2d, mean_pred), 5)
train_mae_data_b1 = np.round(MAE_np(train_Y_2d, upper_pred), 5)
train_mae_data_b2 = np.round(MAE_np(train_Y_2d, lower_pred), 5)
valid_mae_data_b0 = np.round(MAE_np(valid_Y_2d, mean_pred_valid), 5)
valid_mae_data_b1 = np.round(MAE_np(valid_Y_2d, upper_pred_valid), 5)
valid_mae_data_b2 = np.round(MAE_np(valid_Y_2d, lower_pred_valid), 5)

print("====================MAPE======================")
print("train_mape_data_b0:", train_mape_data_b0)
print("train_mape_data_b1:", train_mape_data_b1)
print("train_mape_data_b2:", train_mape_data_b2)
print("valid_mape_data_b0:", valid_mape_data_b0)
print("valid_mape_data_b1:", valid_mape_data_b1)
print("valid_mape_data_b2:", valid_mape_data_b2)

print("\n====================MAE======================")
print("train_mae_data_b0:", train_mae_data_b0)
print("train_mae_data_b1:", train_mae_data_b1)
print("train_mae_data_b2:", train_mae_data_b2)
print("valid_mae_data_b0:", valid_mae_data_b0)
print("valid_mae_data_b1:", valid_mae_data_b1)
print("valid_mae_data_b2:", valid_mae_data_b2)
