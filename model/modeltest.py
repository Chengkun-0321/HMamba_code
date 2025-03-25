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

# ---------------------------
# 修改 MambaBlock 以接受 name 參數
# ---------------------------
class MambaBlock(layers.Layer):
    def __init__(self, modelargs: ModelArgs, name=None):
        super().__init__(name=name)
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
# Self-Attention 模組
# ---------------------------
class SelfAttention(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = layers.Dense(dim)
        self.k_proj = layers.Dense(dim)
        self.v_proj = layers.Dense(dim)
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
# SSMCausalConvBlock 模組
# ---------------------------
class SSMCausalConvBlock(layers.Layer):
    def __init__(self, modelargs: ModelArgs):
        super().__init__()
        self.args = modelargs
        self.A_log = tf.Variable(tf.random.normal([modelargs.model_input_dims, modelargs.model_states]))
        self.D = tf.Variable(tf.ones([modelargs.model_input_dims]))
        self.x_projection = layers.Dense(modelargs.model_states * 2 + modelargs.model_input_dims, use_bias=False)
        self.delta_t_projection = layers.Dense(modelargs.model_input_dims, use_bias=True)
        self.conv1 = layers.Conv1D(filters=modelargs.model_input_dims, kernel_size=3, padding='causal', activation='relu')
        self.linear_filter = layers.Dense(modelargs.model_input_dims, activation='relu')
        
    def call(self, x, h_t_minus_1):
        A = -tf.exp(self.A_log)
        D = self.D
        x_dbl = self.x_projection(x)
        delta, B, C = tf.split(x_dbl, [self.args.model_input_dims, self.args.model_states, self.args.model_states], axis=-1)
        delta = tf.nn.softplus(self.delta_t_projection(delta))
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
# 建立只包含中間分支（mean）的模型
# ---------------------------
def build_middle_model(args: ModelArgs):
    input_layer = layers.Input(shape=(9, 9, 1), name='input')
    seq_length = 9 * 9
    x = layers.Reshape((seq_length, 1))(input_layer)
    x = layers.Dense(args.model_input_dims)(x)
    # 使用命名的中間分支
    middle_block = MambaBlock(args, name="middle_block")
    middle_output = middle_block(x)
    middle_flat = layers.Flatten()(middle_output)
    mean_pred = layers.Dense(args.num_classes, activation=args.final_activation, name="mean")(middle_flat)
    model = Model(inputs=input_layer, outputs=mean_pred, name="middle_model")
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model

# ---------------------------
# 建立完整模型（包含中間、上界、下界分支）
# ---------------------------
def build_full_model(args: ModelArgs):
    input_layer = layers.Input(shape=(9, 9, 1), name='input')
    seq_length = 9 * 9
    x = layers.Reshape((seq_length, 1))(input_layer)
    x = layers.Dense(args.model_input_dims)(x)
    
    # 中間分支
    middle_block = MambaBlock(args, name="middle_block")
    middle_output = middle_block(x)
    
    # 上界分支：Self-Attention + MambaBlock
    attention_upper = SelfAttention(args.model_input_dims)(middle_output, x)
    upper_input = layers.Add(name="upper_input")([x, attention_upper])
    upper_block = MambaBlock(args, name="upper_block")
    upper_output = upper_block(upper_input)
    
    # 下界分支：Self-Attention + MambaBlock
    attention_lower = SelfAttention(args.model_input_dims)(middle_output, x)
    lower_input = layers.Add(name="lower_input")([x, attention_lower])
    lower_block = MambaBlock(args, name="lower_block")
    lower_output = lower_block(lower_input)
    
    middle_flat = layers.Flatten()(middle_output)
    upper_flat = layers.Flatten()(upper_output)
    lower_flat = layers.Flatten()(lower_output)
    
    mean_pred = layers.Dense(args.num_classes, activation=args.final_activation, name="mean")(middle_flat)
    upper_pred = layers.Dense(args.num_classes, activation=args.final_activation, name="upper")(upper_flat)
    lower_pred = layers.Dense(args.num_classes, activation=args.final_activation, name="lower")(lower_flat)
    
    # 串接三個分支的預測 (最終輸出 shape: (batch, 3))
    combined_output = layers.Concatenate(axis=-1, name="combined_output")([mean_pred, upper_pred, lower_pred])
    
    model = Model(inputs=input_layer, outputs=combined_output, name="full_model")
    model.compile(loss=loss_mse, optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model

# ---------------------------
# 自訂 Loss 函數（保持原有版本，此處不再修改）
# ---------------------------
def loss_mse(y_true, y_pred):
    # y_pred shape: (batch, 3)
    mean_pred  = y_pred[:, 0:1]
    upper_pred = y_pred[:, 1:2]
    lower_pred = y_pred[:, 2:3]
    
    theta = 1.0  # 固定值，此處可根據需要動態調整
    loss_total = 0.0
    # Mean 分支 loss
    l_mean = tf.reduce_mean(tf.square(y_true - mean_pred))
    loss_total += l_mean

    # Upper 分支 loss
    diff_upper = y_true - 65.0
    temp_upper = tf.where(diff_upper < theta, 0.0, diff_upper)
    lambda_base_upper = tf.where(diff_upper > theta, 1.0, temp_upper)
    processed_upper = tf.where(y_true >= 70.0, tf.ones_like(temp_upper), tf.math.tanh(temp_upper))
    total_lambda_upper = lambda_base_upper + processed_upper
    l_upper = loss_mse_lambda(upper_pred, y_true, total_lambda_upper)
    l_upper += tf.clip_by_value(loss_mse_lambda_2(upper_pred, mean_pred, y_true, total_lambda_upper),
                                 clip_value_min=0.0, clip_value_max=10000.0)
    loss_total += l_upper

    # Lower 分支 loss
    diff_lower = 65.0 - y_true
    temp_lower = tf.where(diff_lower < theta, 0.0, diff_lower)
    lambda_base_lower = tf.where(diff_lower > theta, 1.0, temp_lower)
    processed_lower = tf.where(y_true <= 60.0, tf.ones_like(temp_lower), tf.math.tanh(temp_lower))
    total_lambda_lower = lambda_base_lower + processed_lower
    l_lower = loss_mse_lambda(lower_pred, y_true, total_lambda_lower)
    loss_total += l_lower

    return loss_total

def loss_mse_lambda(outputs, targets, lam):
    diff = outputs - targets
    loss = tf.reduce_sum(lam * tf.square(diff))
    nonzero = tf.reduce_sum(tf.cast(lam > 0, tf.float32))
    nonzero = tf.maximum(nonzero, 1.0)
    return loss / nonzero

def loss_mse_lambda_2(outputs, outputs2, targets, lam):
    diff = tf.square(outputs - targets) - tf.square(outputs2 - targets)
    loss = tf.reduce_sum(lam * diff)
    nonzero = tf.reduce_sum(tf.cast(lam > 0, tf.float32))
    nonzero = tf.maximum(nonzero, 1.0)
    return loss / nonzero

# ---------------------------
# 資料載入（請確認 npy 檔案路徑與形狀正確）
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


args = ModelArgs(
    model_input_dims=128,
    model_states=32,
    num_layers=5,
    dropout_rate=0.2,
    num_classes=1,
    loss='mse'
)

middle_model = build_middle_model(args)
middle_model.summary()


middle_history = middle_model.fit(train_X_2d, train_Y_2d, batch_size=100, epochs=5,
                                  validation_data=(valid_X_2d, valid_Y_2d))

# 儲存中間模型權重（也可用 model.save_weights）
trained_middle_weights = middle_model.get_layer("middle_block").get_weights()


full_model = build_full_model(args)
full_model.summary()

# 複製中間分支權重到 full_model 中的上界與下界的 MambaBlock 層
upper_block = full_model.get_layer("upper_block")
lower_block = full_model.get_layer("lower_block")
# 將中間分支的權重複製到上界與下界分支
upper_block.set_weights(trained_middle_weights)
lower_block.set_weights(trained_middle_weights)


full_history = full_model.fit(train_X_2d, train_Y_2d, batch_size=100, epochs=5,
                              validation_data=(valid_X_2d, valid_Y_2d))


pred_y = full_model.predict(train_X_2d)
print("Training Predictions (full model):", pred_y)
