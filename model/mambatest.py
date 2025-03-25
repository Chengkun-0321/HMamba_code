import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt

# 為方便除錯，開啟 eager execution（正式訓練時可關閉）
tf.config.run_functions_eagerly(True)

# ---------------------------
# 定義模型參數
# ---------------------------
@dataclass
class ModelArgs:
    model_input_dims: int = 128   # 輸入特徵維度
    model_states: int = 32        # 隱藏狀態維度
    num_layers: int = 5           # 模型層數（若有需要）
    dropout_rate: float = 0.2     # dropout 比例（若有需要）
    num_classes: int = 1          # 輸出單一數值（例如：迴歸預測）
    loss: str = 'mse'             # 這邊不直接使用字串，而是用自訂 loss
    final_activation = None       # 最後一層激活函數

# ---------------------------
# Self-Attention 模組
# Q 從中間層取得，K 與 V 從原始資料取得
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
# 此模組內部進行狀態更新計算，
# 為避免數值爆炸，對中間的指數運算做 clipping 處理，
# 並在最後對狀態更新結果再做 clipping。
# ---------------------------
class SSMCausalConvBlock(layers.Layer):
    def __init__(self, modelargs: ModelArgs):
        super().__init__()
        self.args = modelargs
        # SSM 參數（隨機初始化，僅供示範）
        self.A_log = tf.Variable(tf.random.normal([modelargs.model_input_dims, modelargs.model_states]), trainable=True)
        self.D = tf.Variable(tf.ones([modelargs.model_input_dims]), trainable=True)
        # 將輸入 x 投影至三組參數
        self.x_projection = layers.Dense(modelargs.model_states * 2 + modelargs.model_input_dims, use_bias=False)
        self.delta_t_projection = layers.Dense(modelargs.model_input_dims, use_bias=True)
        # 因果卷積（使用 causal padding 以保持時間順序）
        self.conv1 = layers.Conv1D(filters=modelargs.model_input_dims, kernel_size=3, padding='causal', activation='relu')
        # SSM 輸出層
        self.linear_filter = layers.Dense(modelargs.model_input_dims, activation='relu')
        
    def call(self, x, h_t_minus_1):
        # A 以負指數形式呈現
        A = -tf.exp(self.A_log)
        D = self.D
        x_dbl = self.x_projection(x)
        # 分割成 delta、B、C 三部分
        delta, B, C = tf.split(x_dbl, [self.args.model_input_dims, self.args.model_states, self.args.model_states], axis=-1)
        delta = tf.nn.softplus(self.delta_t_projection(delta))
        # 計算新的隱藏狀態
        h_t = self.selective_scan(h_t_minus_1, delta, A, B, C, D)
        ssm_output = self.linear_filter(h_t)
        causal_output = self.conv1(ssm_output)
        fused_output = ssm_output * causal_output
        return fused_output, h_t
        
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
# MambaBlock 模組
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
# 定義完整模型結構
# 輸出 shape 為 (batch, 3)，分別代表 [mean, upper, lower]
# ---------------------------
def build_model(args: ModelArgs):
    input_layer = layers.Input(shape=(9, 9, 1), name='input')
    seq_length = 9 * 9  # 81
    # 攤平成 (batch, 81, 1)
    x = layers.Reshape((seq_length, 1))(input_layer)
    # 投影到高維空間
    x = layers.Dense(args.model_input_dims)(x)
    
    # Middle branch (mean)
    middle_output = MambaBlock(args)(x)
    
    # Upper branch：使用 Self-Attention，query 來自 middle_output，K/V 來自原始 x
    attention_upper = SelfAttention(args.model_input_dims)(middle_output, x)
    upper_input = layers.Add()([x, attention_upper])
    upper_output = MambaBlock(args)(upper_input)
    
    # Lower branch：同理，使用 middle_output 作為 query
    attention_lower = SelfAttention(args.model_input_dims)(middle_output, x)
    lower_input = layers.Add()([x, attention_lower])
    lower_output = MambaBlock(args)(lower_input)
    
    # 將三個分支攤平成一維向量
    middle_flat = layers.Flatten()(middle_output)
    upper_flat = layers.Flatten()(upper_output)
    lower_flat = layers.Flatten()(lower_output)
    
    # 分別產生各分支預測值
    mean_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='mean')(middle_flat)
    upper_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='upper')(upper_flat)
    lower_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='lower')(lower_flat)
    
    # 串接成最終輸出 (batch, 3)
    combined_output = layers.Concatenate(axis=-1)([mean_pred, upper_pred, lower_pred])
    
    model = Model(inputs=input_layer, outputs=combined_output, name='Mamba_Virtual_Measurement')
    
    # 使用較低學習率並加入梯度裁剪，防止梯度爆炸
    adam = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    #adam = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(loss=loss_mse, optimizer=adam)
    return model

# ---------------------------
# 自訂 Loss 函數
# ---------------------------
def loss_mse_lambda(outputs, targets, lam):
    # outputs, targets, lam 均為 shape (batch, 1)
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

def loss_mse(y_true, y_pred):
    """
    y_pred: Tensor shape (batch, 3)
      索引 0: mean 預測
      索引 1: upper 預測
      索引 2: lower 預測
    """
    # 拆分各分支預測
    mean_pred  = y_pred[:, 0:1]
    upper_pred = y_pred[:, 1:2]
    lower_pred = y_pred[:, 2:3]
    
    theta = 1.0
    loss_total = 0.0
    
    # Mean 分支的 MSE loss
    l_mean = tf.reduce_mean(tf.square(y_true - mean_pred))
    loss_total += l_mean

    # Upper 分支：當 y_true 大於 65 時給予額外懲罰
    diff_upper = y_true - 65.0
    temp_upper = tf.where(diff_upper < theta, 0.0, diff_upper)
    lambda_base_upper = tf.where(diff_upper > theta, 1.0, temp_upper)
    processed_upper = tf.where(y_true >= 70.0, tf.ones_like(temp_upper), tf.math.tanh(temp_upper))
    total_lambda_upper = lambda_base_upper + processed_upper
    l_upper = loss_mse_lambda(upper_pred, y_true, total_lambda_upper)
    l_upper += tf.clip_by_value(loss_mse_lambda_2(upper_pred, mean_pred, y_true, total_lambda_upper),
                                 clip_value_min=0.0, clip_value_max=10000.0)
    loss_total += l_upper

    # Lower 分支：當 y_true 小於 65 時給予額外懲罰
    diff_lower = 65.0 - y_true
    temp_lower = tf.where(diff_lower < theta, 0.0, diff_lower)
    lambda_base_lower = tf.where(diff_lower > theta, 1.0, temp_lower)
    processed_lower = tf.where(y_true <= 60.0, tf.ones_like(temp_lower), tf.math.tanh(temp_lower))
    total_lambda_lower = lambda_base_lower + processed_lower
    l_lower = loss_mse_lambda(lower_pred, y_true, total_lambda_lower)
    loss_total += l_lower

    return loss_total

# ---------------------------
# 載入資料（請確認 npy 檔案路徑與資料形狀正確）
# 假設資料形狀為 (num_samples, 9, 9)
# ---------------------------
# 請根據實際路徑調整下列 np.load() 的參數
CP_data_train_2d_X = np.load('./cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_train_2d_Y = np.load('./cnn-2d_2020-09-09_11-45-24_y.npy')
CP_data_train_2d_X_tensor = tf.convert_to_tensor(CP_data_train_2d_X, dtype=tf.float32)
CP_data_train_2d_Y_tensor = tf.convert_to_tensor(CP_data_train_2d_Y, dtype=tf.float32)
# 若原始資料 shape 為 (num_samples, 9, 9)，則擴展一個 channel 維度
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
# 初始化模型並開始訓練
# ---------------------------
args = ModelArgs(
    model_input_dims=128,
    model_states=32,
    num_layers=5,
    dropout_rate=0.2,
    num_classes=1,
    loss=loss_mse  # 此參數在 build_model 中實際上不直接使用
)

model = build_model(args)
model.summary()

# 訓練設定（可根據需求調整 epochs 與 batch_size）
epochs = 5
history = model.fit(
    train_X_2d, train_Y_2d,
    batch_size=100,
    epochs=epochs,
    validation_data=(valid_X_2d, valid_Y_2d),
    validation_freq=1
)

# ---------------------------
# 預測與評估
# ---------------------------
pred_y = model.predict(train_X_2d)
pred_valid_y = model.predict(valid_X_2d)
print("模型目標:", train_Y_2d)
print("Training Predictions:", pred_y)

# 計算 MAPE 與 MAE（僅作簡單評估）
def mape_np(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MAE_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

mean_pred  = pred_y[:, 0:1]
upper_pred = pred_y[:, 1:2]
lower_pred = pred_y[:, 2:3]

train_mape_mean = np.round(mape_np(train_Y_2d, mean_pred), 5)
train_mape_upper = np.round(mape_np(train_Y_2d, upper_pred), 5)
train_mape_lower = np.round(mape_np(train_Y_2d, lower_pred), 5)

print("==================== Training MAPE ======================")
print("Mean:", train_mape_mean)
print("Upper:", train_mape_upper)
print("Lower:", train_mape_lower)

# 根據條件選擇預測分支，這裡僅作示例
mean_range_upper = 66.0
mean_range_lower = 64.0
pred_train_net_combine = np.copy(pred_y[:, 0])  # 預設使用 mean 分支

upper_train_gt, upper_train_pred, lower_train_gt, lower_train_pred = [], [], [], []

for idx, y in enumerate(train_Y_2d):
    # 若在 eager mode 下，每筆 y 都可直接轉成 numpy 數值
    y_val = y.numpy() if hasattr(y, 'numpy') else y
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

upper_train_mape_data = np.round(mape_np(upper_train_pred, upper_train_gt), 5) if upper_train_gt.size > 0 else None
lower_train_mape_data = np.round(mape_np(lower_train_pred, lower_train_gt), 5) if lower_train_gt.size > 0 else None

upper_train_mae_data = np.round(MAE_np(upper_train_pred, upper_train_gt), 5) if upper_train_gt.size > 0 else None
lower_train_mae_data = np.round(MAE_np(lower_train_pred, lower_train_gt), 5) if lower_train_gt.size > 0 else None

print("\n=================== 上下界 MAPE ======================")
print("upper_train_mape_data:", upper_train_mape_data)
print("lower_train_mape_data:", lower_train_mape_data)
print("\n=================== 上下界 MAE ======================")
print("upper_train_mae_data:", upper_train_mae_data)
print("lower_train_mae_data:", lower_train_mae_data)
