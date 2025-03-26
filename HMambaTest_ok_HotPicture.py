import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
import os
import argparse
import csv
import seaborn as sns
from tqdm import tqdm

# --------------------------------
# 定義命令列參數（可自行修改預設值）
# --------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Mamba Virtual Measurement - 參數設定")
    parser.add_argument("--test_x_path", type=str,
                        default='./testing_data/cnn-2d_2020-09-09_11-45-24_x.npy',
                        help="測試資料 X 的路徑")
    parser.add_argument("--test_y_path", type=str,
                        default='./testing_data/cnn-2d_2020-09-09_11-45-24_y.npy',
                        help="測試資料 Y 的路徑")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoint/123.hdf5",
                        help="模型 checkpoint 的路徑")
    parser.add_argument("--mean", type=float, default=65.0,
                        help="mean 數值")
    parser.add_argument("--boundary_upper", type=float, default=70.0,
                        help="上邊界數值")
    parser.add_argument("--boundary_lower", type=float, default=60.0,
                        help="下邊界數值")
    return parser.parse_args()

# --------------------------------
# 定義模型參數
# --------------------------------
@dataclass
class ModelArgs:
    model_input_dims: int = 128   # 每個 token 投影後的維度
    model_states: int = 32        # 與原架構相容，此處未實際使用
    num_layers: int = 5           # 固定為 5
    dropout_rate: float = 0.2     # (此範例中未使用，可自行添加)
    num_classes: int = 1          # 輸出單位數，迴歸任務通常為 1
    loss: str = 'mse'             # 此處不會用字串，而是會傳入自定義 loss 函數
    final_activation = None       # 迴歸任務不使用激活函數

# --------------------------------
# 定義 Self-Attention 模組（更新：可回傳注意力權重）
# --------------------------------
class SelfAttention(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q_proj = layers.Dense(dim)
        self.k_proj = layers.Dense(dim)
        self.v_proj = layers.Dense(dim)
        self.softmax = layers.Softmax(axis=-1)
        self.out_proj = layers.Dense(dim)

    # 新增 return_attention 參數，若 True 則回傳 (output, attention_scores)
    def call(self, query, key_value, return_attention=False):
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        attention_scores = self.softmax(tf.matmul(Q, K, transpose_b=True))
        attention_output = tf.matmul(attention_scores, V)
        output = self.out_proj(attention_output)
        if return_attention:
            return output, attention_scores
        else:
            return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim  # 把層的初始化參數保存下來
        })
        return config


# --------------------------------
# 定義 SSMCausalConvBlock 模組
# --------------------------------
class SSMCausalConvBlock(layers.Layer):
    def __init__(self, modelargs: ModelArgs):
        super().__init__()
        self.args = modelargs
        self.modelargs = modelargs
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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "modelargs": self.args.__dict__  # 將 ModelArgs 轉為字典格式
        })
        return config

# --------------------------------
# 定義 MambaBlock（使用 SSMCausalConvBlock）
# --------------------------------
class MambaBlock(layers.Layer):
    def __init__(self, modelargs: ModelArgs, **kwargs):
        super().__init__(**kwargs)
        self.modelargs = modelargs  # 儲存參數
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
        h0 = x_conv  # 使用 conv1 輸出作為初始隱藏狀態
        ssm_out, _ = self.ssm_block(x_conv, h0)
        return self.out_proj(ssm_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "modelargs": self.modelargs.__dict__
        })
        return config
"""class MambaBlock(layers.Layer):
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
        return self.out_proj(ssm_out)"""

# --------------------------------
# 定義完整模型結構 (Mamba Virtual Measurement)
# 並加入 return_attention 參數，使模型可以回傳注意力圖 att1 與 att2
# --------------------------------
def build_model(args: ModelArgs, return_attention=False):
    input_layer = layers.Input(shape=(9, 9, 1), name='input')
    seq_length = 9 * 9
    x = layers.Reshape((seq_length, 1))(input_layer)
    x = layers.Dense(args.model_input_dims)(x)
    
    # Middle branch (mean)
    middle_output = MambaBlock(args)(x)
    
    # Upper branch：利用帶有注意力回傳的 SelfAttention 層
    upper_sa = SelfAttention(args.model_input_dims)
    if return_attention:
        attention_upper_out, attn_upper_scores = upper_sa(middle_output, x, return_attention=True)
    else:
        attention_upper_out = upper_sa(middle_output, x)
    upper_input = layers.Add()([x, attention_upper_out])
    upper_output = MambaBlock(args)(upper_input)
    
    # Lower branch：同理
    lower_sa = SelfAttention(args.model_input_dims)
    if return_attention:
        attention_lower_out, attn_lower_scores = lower_sa(middle_output, x, return_attention=True)
    else:
        attention_lower_out = lower_sa(middle_output, x)
    lower_input = layers.Add()([x, attention_lower_out])
    lower_output = MambaBlock(args)(lower_input)
    
    middle_flat = layers.Flatten()(middle_output)
    upper_flat = layers.Flatten()(upper_output)
    lower_flat = layers.Flatten()(lower_output)
    
    mean_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='mean')(middle_flat)
    upper_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='upper')(upper_flat)
    lower_pred = layers.Dense(args.num_classes, activation=args.final_activation, name='lower')(lower_flat)
    
    if return_attention:
        # 由於注意力分數的 shape 為 (batch, 81, 81)，
        # 這裡取出每個 sample 中第一個 query token 的注意力（shape: (batch, 81)），
        # 再 reshape 為 (batch, 1, 9, 9) 供後續熱圖使用
        att_upper_map = layers.Lambda(lambda t: tf.reshape(t[:, 0, :], (-1, 1, 9, 9)), name='att_upper')(attn_upper_scores)
        att_lower_map = layers.Lambda(lambda t: tf.reshape(t[:, 0, :], (-1, 1, 9, 9)), name='att_lower')(attn_lower_scores)
        outputs = [mean_pred, upper_pred, lower_pred, att_upper_map, att_lower_map]
    else:
        outputs = [mean_pred, upper_pred, lower_pred]
    
    model = Model(inputs=input_layer, outputs=outputs, name='Mamba_Virtual_Measurement')
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    
    if return_attention:
        # 當附加輸出注意力圖時，設定 dummy loss 使其不影響訓練
        def dummy_loss(y_true, y_pred):
            return tf.zeros_like(y_pred)
        model.compile(loss=[loss_mse, dummy_loss, dummy_loss, dummy_loss, dummy_loss], optimizer=adam)
    else:
        model.compile(loss=loss_mse, optimizer=adam)
    
    return model

# --------------------------------
# 自定義 Loss 函數（全部改為純 TensorFlow 運算，不使用 .numpy()）
# --------------------------------
def loss_mse(y_true, y_pred):
    theta = 1.0
    total_loss = 0.0
    # 分支 0：mean 直接用 MSE
    l0 = MSE(y_pred[0], y_true) if isinstance(y_pred, list) else MSE(y_pred, y_true)
    total_loss += l0
    # 分支 1：upper
    raw = y_true - 65.0
    lambda_val = tf.where(raw < theta, 0.0, raw)
    lambda_base = tf.where(lambda_val > theta, 1.0, lambda_val)
    lambda_modified = tf.where(tf.equal(lambda_val, 0.0),
                               0.0,
                               tf.where(y_true >= 70.0, 1.0, tf.math.tanh(lambda_val)))
    total_lambda = lambda_base + lambda_modified
    l1 = loss_mse_lambda(y_pred[1] if isinstance(y_pred, list) else y_pred, y_true, total_lambda)
    total_loss += l1
    # 分支 2：lower
    raw = 65.0 - y_true
    lambda_val = tf.where(raw < theta, 0.0, raw)
    lambda_base = tf.where(lambda_val > theta, 1.0, lambda_val)
    lambda_modified = tf.where(tf.equal(lambda_val, 0.0),
                               0.0,
                               tf.where(y_true <= 60.0, 1.0, tf.math.tanh(lambda_val)))
    total_lambda = lambda_base + lambda_modified
    l2 = loss_mse_lambda(y_pred[2] if isinstance(y_pred, list) else y_pred, y_true, total_lambda)
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

# --------------------------------
# 載入訓練與驗證資料（這部分路徑固定，可依需求自行改）
# --------------------------------
CP_data_train_2d_X = np.load('./training_data/PETBottle/cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_train_2d_Y = np.load('./training_data/PETBottle/cnn-2d_2020-09-09_11-45-24_y.npy')
CP_data_train_2d_X_tensor = tf.convert_to_tensor(CP_data_train_2d_X)
CP_data_train_2d_Y_tensor = tf.convert_to_tensor(CP_data_train_2d_Y)
train_X_2d = tf.expand_dims(CP_data_train_2d_X_tensor, axis=3)
train_Y_2d = CP_data_train_2d_Y_tensor

CP_data_valid_2d_X = np.load('./validation_data/PETBottle/cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_valid_2d_Y = np.load('./validation_data/PETBottle/cnn-2d_2020-09-09_11-45-24_y.npy')
CP_data_valid_2d_X_tensor = tf.convert_to_tensor(CP_data_valid_2d_X)
CP_data_valid_2d_Y_tensor = tf.convert_to_tensor(CP_data_valid_2d_Y)
valid_X_2d = tf.expand_dims(CP_data_valid_2d_X_tensor, axis=3)
valid_Y_2d = CP_data_valid_2d_Y_tensor

# --------------------------------
# 定義熱圖繪製函數
# --------------------------------
def heatmap_drawer(attention, gt_value, pred_value, pred_cls, maps, result_path):
    """
    繪製注意力熱圖並儲存至 result_path 中的 AN_U 與 AN_L 子資料夾
    attention: list，包含兩個 numpy 陣列，分別為上邊界與下邊界注意力圖，形狀預期 [num_samples, 1, H, W]
    gt_value: 真實值 numpy 陣列，形狀 [num_samples,]
    pred_value: 預測值 numpy 陣列，形狀 [num_samples,]
    pred_cls: 預測分類 numpy 陣列，形狀 [num_samples,]，上邊界應為 1，下邊界應為 2
    maps: y 軸標籤 list
    result_path: 結果儲存的根目錄
    """
    # squeeze 掉注意力圖第二維度
    an_u = np.squeeze(attention[0], axis=1)  # 上邊界注意力圖, shape: (num_samples, H, W)
    an_l = np.squeeze(attention[1], axis=1)  # 下邊界注意力圖, shape: (num_samples, H, W)
    
    # 建立結果資料夾
    upper_path = os.path.join(result_path, "AN_U")
    lower_path = os.path.join(result_path, "AN_L")
    os.makedirs(upper_path, exist_ok=True)
    os.makedirs(lower_path, exist_ok=True)
    
    # 上邊界熱圖繪製
    for idx, att_map in enumerate(tqdm(an_u, desc="Upper branch heatmaps")):
        _gt_value = np.round(gt_value[idx], 2)
        _pred_value = np.round(pred_value[idx], 2)
        if pred_cls[idx] != 1:  # 僅針對分類為 1 的上邊界樣本
            continue
        att_map_copy = att_map.copy()
        if att_map_copy.shape[0] > 0:
            att_map_copy[0, 1:] = -1
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(att_map_copy, xticklabels=list("123456789"), yticklabels=maps,
                         linewidths=0.1, cbar=True, square=False)
        title = f"{idx}_gt-{_gt_value}_pred-{_pred_value}"
        plt.title(title, fontsize="x-large")
        plt.ylabel("Process Data", fontsize="x-large")
        # 調整左側留白以顯示完整標籤
        plt.subplots_adjust(left=0.3)
        plt.savefig(os.path.join(upper_path, f"{title}.svg"))
        plt.close()
    
    # 下邊界熱圖繪製
    for idx, att_map in enumerate(tqdm(an_l, desc="Lower branch heatmaps")):
        _gt_value = np.round(gt_value[idx], 2)
        _pred_value = np.round(pred_value[idx], 2)
        if pred_cls[idx] != 2:  # 僅針對分類為 2 的下邊界樣本
            continue
        att_map_copy = att_map.copy()
        if att_map_copy.shape[0] > 0:
            att_map_copy[0, 1:] = -1
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(att_map_copy, xticklabels=list("123456789"), yticklabels=maps,
                         linewidths=0.1, cbar=True, square=False)
        title = f"{idx}_gt-{_gt_value}_pred-{_pred_value}"
        plt.title(title, fontsize="x-large")
        plt.ylabel("Process Data", fontsize="x-large")
        plt.subplots_adjust(left=0.3)
        plt.savefig(os.path.join(lower_path, f"{title}.svg"))
        plt.close()

# --------------------------------
# 主程式區塊：載入測試資料、checkpoint 與其他參數，進行模型預測與結果評估，同時繪製熱圖
# --------------------------------
if __name__ == '__main__':
    args_cli = parse_args()
    
    CP_data_test_2d_X = np.load(args_cli.test_x_path)
    CP_data_test_2d_Y = np.load(args_cli.test_y_path)
    CP_data_test_2d_X_tensor = tf.convert_to_tensor(CP_data_test_2d_X)
    CP_data_test_2d_Y_tensor = tf.convert_to_tensor(CP_data_test_2d_Y)
    test_X_2d = tf.expand_dims(CP_data_test_2d_X_tensor, axis=3)
    test_Y_2d = CP_data_test_2d_Y_tensor 

    checkpoint_path = args_cli.checkpoint_path
    mean = args_cli.mean
    boundary_upper = args_cli.boundary_upper
    boundary_lower = args_cli.boundary_lower

    theta = 2.0
    mean_range_upper = mean + theta
    mean_range_lower = mean - theta

    model_args = ModelArgs(
        model_input_dims=128,
        model_states=32,
        num_layers=5,
        dropout_rate=0.2,
        num_classes=1,
        loss=loss_mse
    )
    # 設定 return_attention=True 以便於取得注意力圖供熱圖繪製
    model1 = build_model(model_args, return_attention=True)
    # 嘗試載入權重，使用 skip_mismatch 避免錯誤
    print("🔍 嘗試載入已訓練權重...")
    #model1.load_weights(checkpoint_path)  # 先嘗試不跳過 mismatch
    model1.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    
    # 檢查 MambaBlock 層的權重是否與 .h5 權重檔匹配
    print("\n🔍 檢查 MambaBlock 權重載入情況：")
    for layer in model1.layers:
        if "mamba_block" in layer.name:
            print(f"Layer: {layer.name}")
            for weight in layer.weights:
                print(f"- {weight.name}: {weight.shape}, mean: {tf.reduce_mean(weight).numpy():.6f}, std: {tf.math.reduce_std(weight).numpy():.6f}")
    
    # 顯示完整模型架構
    print("\n🔍 模型架構總覽：")
    model1.summary()

    # 使用模型預測，pred_y 依序為 [mean_pred, upper_pred, lower_pred, att_upper_map, att_lower_map]
    pred_y = model1.predict(test_X_2d)
    mean_pred = pred_y[0]
    print("mean_pred shape:", mean_pred.shape)

    # ---------------------- 利用 mean 分支預測作為分類器 ----------------------
    pred_test_net_cls = np.copy(mean_pred)
    for idx in range(pred_test_net_cls.shape[0]):
        data = pred_test_net_cls[idx, 0]
        if mean_range_lower <= data <= mean_range_upper:
            pred_test_net_cls[idx, 0] = 0
        elif data > mean_range_upper:
            pred_test_net_cls[idx, 0] = 1
        elif data < mean_range_lower:
            pred_test_net_cls[idx, 0] = 2
    pred_test_net_cls = np.squeeze(pred_test_net_cls).astype('int64')

    # 利用分類結果整合 RACNN 與 Classifier 預測
    pred_test_net_combine = np.copy(mean_pred)
    for idx, cls_result in enumerate(pred_test_net_cls):
        # 若分類結果為 0 使用 mean_pred，1 使用 upper_pred，2 使用 lower_pred
        pred_test_net_combine[idx, 0] = pred_y[cls_result][idx, 0]
    pred_test_net_combine = tf.convert_to_tensor(pred_test_net_combine)

    # ---------------------- BN0 混淆矩陣 (使用 mean 分支預測) ----------------------
    confusion_matrix = np.zeros((3, 3))
    for idx in range(mean_pred.shape[0]):
        gt_y_value = test_Y_2d[idx].numpy()
        pred_y_value = mean_pred[idx, 0]
        if mean_range_lower <= gt_y_value <= mean_range_upper:
            if mean_range_lower <= pred_y_value <= mean_range_upper:
                confusion_matrix[0, 0] += 1
            elif (mean_range_upper < pred_y_value < boundary_upper) or (mean_range_lower > pred_y_value > boundary_lower):
                confusion_matrix[0, 1] += 1
            elif pred_y_value >= boundary_upper or pred_y_value <= boundary_lower:
                confusion_matrix[0, 2] += 1
            else:
                print("Data似乎有問題0。")
        elif (mean_range_upper < gt_y_value < boundary_upper) or (mean_range_lower > gt_y_value > boundary_lower):
            if mean_range_lower <= pred_y_value <= mean_range_upper:
                confusion_matrix[1, 0] += 1
            elif (mean_range_upper < pred_y_value < boundary_upper) or (mean_range_lower > pred_y_value > boundary_lower):
                confusion_matrix[1, 1] += 1
            elif pred_y_value >= boundary_upper or pred_y_value <= boundary_lower:
                confusion_matrix[1, 2] += 1
            else:
                print("Data似乎有問題1。")
        elif gt_y_value >= boundary_upper or gt_y_value <= boundary_lower:
            if mean_range_lower <= pred_y_value <= mean_range_upper:
                confusion_matrix[2, 0] += 1
            elif (mean_range_upper < pred_y_value < boundary_upper) or (mean_range_lower > pred_y_value > boundary_lower):
                confusion_matrix[2, 1] += 1
            elif pred_y_value >= boundary_upper or pred_y_value <= boundary_lower:
                confusion_matrix[2, 2] += 1
            else:
                print("Data似乎有問題2。")
        else:
            print("Data似乎有問題3。")
    print("BN0 confusion matrix (mean branch):")
    print(confusion_matrix)

    confusion_matrix_col = np.sum(confusion_matrix, axis=0)
    confusion_matrix_row = np.sum(confusion_matrix, axis=1)

    false_alarm_probability = (confusion_matrix[0,1] + confusion_matrix[0,2] +
                                 confusion_matrix[1,0] + confusion_matrix[1,2] +
                                 confusion_matrix[2,0] + confusion_matrix[2,1]) / np.sum(confusion_matrix_col)
    print(f"False Alarm Probability: {false_alarm_probability}")
    precision = np.array([
        confusion_matrix[0, 0] / confusion_matrix_col[0] if confusion_matrix_col[0] != 0 else 0,
        confusion_matrix[1, 1] / confusion_matrix_col[1] if confusion_matrix_col[1] != 0 else 0,
        confusion_matrix[2, 2] / confusion_matrix_col[2] if confusion_matrix_col[2] != 0 else 0
    ])
    print(f"precision: {precision}")
    recall = np.array([confusion_matrix[0,0]/confusion_matrix_row[0],
                       confusion_matrix[1,1]/confusion_matrix_row[1],
                       confusion_matrix[2,2]/confusion_matrix_row[2]])
    print(f"recall: {recall}")
    accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1] + confusion_matrix[2,2]) / np.sum(confusion_matrix_row)
    print(f"accuracy: {accuracy}")

    # ---------------------- 邊界檢測之陣列與評估函數 ----------------------
    detect_range = 4
    detect_precision = 0.5
    interval_upper = []
    interval_lower = []
    split_interval = int(detect_range / detect_precision)
    for idx in range(1, split_interval + 1):
        interval_upper.append([boundary_upper - 0.5 * idx, boundary_upper + 0.5 * idx, 0.5 * idx])
        interval_lower.append([boundary_lower - 0.5 * idx, boundary_lower + 0.5 * idx, 0.5 * idx])
    interval_upper.append([boundary_upper, boundary_upper + 999.0, 0.0])
    interval_lower.append([boundary_lower - 999.0, boundary_lower, 0.0])

    def MAE(f_x, y):
        num = len(f_x)
        summation = sum(abs(f_x - y))
        result = summation / num
        return result

    def RMSE(f_x, y):
        num = len(f_x)
        summation = sum((f_x - y) ** 2)
        result = (summation / num) ** 0.5
        return result

    def MAPE(y_true, y_pred):
        n = len(y_true)
        mape_val = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
        return mape_val

    # ------------- BN0 邊界檢測 -------------
    interval_upper_detect_result = []
    for idx1, data in enumerate(interval_upper):
        interval_upper_test_data = []
        interval_upper_pred_data = []
        for idx2, y in enumerate(pred_test_net_combine.numpy()):
            if data[0] <= y[0] <= data[1]:
                interval_upper_test_data.append(test_Y_2d[idx2].numpy())
                interval_upper_pred_data.append(mean_pred[idx2, 0])
        if len(interval_upper_test_data) > 0 and len(interval_upper_pred_data) > 0:
            mean_absolute_error = MAE(np.array(interval_upper_test_data), np.array(interval_upper_pred_data))
            root_mean_square_error = RMSE(np.array(interval_upper_test_data), np.array(interval_upper_pred_data))
            mape_val = MAPE(np.array(interval_upper_test_data), np.array(interval_upper_pred_data))
            max_error = np.max(np.abs(np.array(interval_upper_test_data) - np.array(interval_upper_pred_data)))
        else:
            mean_absolute_error, root_mean_square_error, mape_val, max_error = 0.0, 0.0, 0.0, 0.0
        interval_upper_detect_result.append(
                [np.array(data), 
                 np.array(interval_upper_test_data), 
                 np.array(interval_upper_pred_data),
                 np.array([mean_absolute_error, root_mean_square_error, mape_val, max_error])]
            )
    """
    #檢查 interval_upper_detect_result 的內容
    for i, item in enumerate(interval_upper_detect_result):
        print(f"Index {i}: {type(item)}, Length: {len(item) if isinstance(item, list) else 'N/A'}")
    #--------------------------------------
    #確認內部數據結構
    for i, item in enumerate(interval_upper_detect_result):
        print(f"Index {i}: Type: {type(item)}, Length: {len(item)}")
        for j, sub_item in enumerate(item):
            print(f"  ├── Element {j}: Type: {type(sub_item)}, Shape: {sub_item.shape if isinstance(sub_item, np.ndarray) else 'N/A'}")
    #-------------
    """

    max_len = max(max(len(sub_item) for sub_item in item) for item in interval_upper_detect_result)

    # 讓所有元素填充到相同長度
    interval_upper_detect_result = [
        [np.pad(sub_item, (0, max_len - len(sub_item)), constant_values=0) if len(sub_item) < max_len else sub_item
        for sub_item in item]
        for item in interval_upper_detect_result
    ]

    # 轉成 NumPy 陣列
    interval_upper_detect_result = np.array(interval_upper_detect_result)

    interval_lower_detect_result = []
    for idx1, data in enumerate(interval_lower):
        interval_lower_test_data = []
        interval_lower_pred_data = []
        for idx2, y in enumerate(pred_test_net_combine.numpy()):
            if data[0] <= y[0] <= data[1]:
                interval_lower_test_data.append(test_Y_2d[idx2].numpy())
                interval_lower_pred_data.append(mean_pred[idx2, 0])
        if len(interval_lower_test_data) > 0 and len(interval_lower_pred_data) > 0:
            mean_absolute_error = MAE(np.array(interval_lower_test_data), np.array(interval_lower_pred_data))
            root_mean_square_error = RMSE(np.array(interval_lower_test_data), np.array(interval_lower_pred_data))
            mape_val = MAPE(np.array(interval_lower_test_data), np.array(interval_lower_pred_data))
            max_error = np.max(np.abs(np.array(interval_lower_test_data) - np.array(interval_lower_pred_data)))
        else:
            mean_absolute_error, root_mean_square_error, mape_val, max_error = 0.0, 0.0, 0.0, 0.0

        # 確保 interval_lower_test_data 和 interval_lower_pred_data 形狀一致
        interval_lower_test_data = np.array(interval_lower_test_data)  # 確保轉換為 NumPy 陣列
        interval_lower_pred_data = np.array(interval_lower_pred_data)  # 確保轉換為 NumPy 陣列

        interval_lower_test_data = np.squeeze(interval_lower_test_data) if interval_lower_test_data.size > 0 else np.array([0.0])
        interval_lower_pred_data = np.squeeze(interval_lower_pred_data) if interval_lower_pred_data.size > 0 else np.array([0.0])

        # 確保數據為 float
        """
        mean_absolute_error = float(mean_absolute_error)
        root_mean_square_error = float(root_mean_square_error)
        mape_val = float(mape_val)
        max_error = float(max_error)
        """
        # --------------
        # 確保 interval_lower_test_data 和 interval_lower_pred_data 形狀一致
        """
        print(f"interval_lower_test_data shape: {np.array(interval_lower_test_data).shape}")
        print(f"interval_lower_pred_data shape: {np.array(interval_lower_pred_data).shape}")
        """
        # ----------------------------------------------------------------
        # 發現形狀不一致，則可以統一轉換為一維陣列：
        interval_lower_test_data = np.array(interval_lower_test_data).flatten()
        interval_lower_pred_data = np.array(interval_lower_pred_data).flatten()
        # ----------------------------------
        # 確保 mean_absolute_error 等數據為 float
        mean_absolute_error = np.array(mean_absolute_error).flatten()
        root_mean_square_error = np.array(root_mean_square_error).flatten()
        mape_val = np.array(mape_val).flatten()
        max_error = np.array(max_error).flatten()

        mean_absolute_error = float(mean_absolute_error[0]) if mean_absolute_error.size > 0 else 0.0
        root_mean_square_error = float(root_mean_square_error[0]) if root_mean_square_error.size > 0 else 0.0
        mape_val = float(mape_val[0]) if mape_val.size > 0 else 0.0
        max_error = float(max_error[0]) if max_error.size > 0 else 0.0
        # ----------------------------------
        interval_lower_detect_result.append(
                [np.array(data), 
                np.array(interval_lower_test_data), 
                np.array(interval_lower_pred_data),
                np.array([mean_absolute_error, root_mean_square_error, mape_val, max_error])]
            )
    """# 確保所有元素形狀一致
    max_len = max(len(item) for item in interval_lower_detect_result)

    # 讓所有元素填充到相同長度
    interval_lower_detect_result = [
        np.pad(item, (0, max_len - len(item)), constant_values=0) if isinstance(item, np.ndarray) and len(item) < max_len else item
        for item in interval_lower_detect_result
    ]
    # --------------------"""
    """
    # 檢查並確保數據形狀一致
    for i, item in enumerate(interval_lower_detect_result):
        print(f"Index {i}: Type: {type(item)}, Length: {len(item) if hasattr(item, '__len__') else 'N/A'}")
        for j, sub_item in enumerate(item):
            print(f"  ├── Element {j}: Type: {type(sub_item)}, Shape: {sub_item.shape if isinstance(sub_item, np.ndarray) else 'N/A'}")
    """
    # 轉換成 NumPy 物件陣列，允許不同形狀
    interval_lower_detect_result = np.array(interval_lower_detect_result, dtype=object)
    #interval_lower_detect_result = np.array(interval_lower_detect_result)

    with open('result_data/(BN0)bonduary(upper+lower)_detect_result.csv', 'w') as f:
        f.write(f"########,上邊界({boundary_upper}),區間檢測,結果,########\n")
        for idx, (low, up, bound_val) in enumerate(interval_upper):
            f.write(f",({idx+1}) {boundary_upper}±{bound_val} [{low}~{up}]")
        f.write('\nMAE,')
        for (_MAE, _, _, _) in interval_upper_detect_result[:, -1]:
            f.write(f"{_MAE},")
        f.write('\nRMSE,')
        for (_, _RMSE, _, _) in interval_upper_detect_result[:, -1]:
            f.write(f"{_RMSE},")
        f.write('\nMAPE,')
        for (_, _, _MAPE, _) in interval_upper_detect_result[:, -1]:
            f.write(f"{_MAPE},")
        f.write('\nME,')
        for (_, _, _, _ME) in interval_upper_detect_result[:, -1]:
            f.write(f"{_ME},")
        for interval, __, _, (_MAE, _RMSE, _MAPE, _ME) in interval_upper_detect_result:
            print(interval, _MAE, _RMSE, _MAPE, _ME)
        
        f.write(f"\n\n########,下邊界({boundary_lower}),區間檢測,結果,########\n")
        for idx, (low, up, bound_val) in enumerate(interval_lower):
            f.write(f",({idx+1}) {boundary_lower}±{bound_val} [{low}~{up}]")
        f.write('\nMAE,')
        for (_MAE, _, _, _) in interval_lower_detect_result[:, -1]:
            f.write(f"{_MAE},")
        f.write('\nRMSE,')
        for (_, _RMSE, _, _) in interval_lower_detect_result[:, -1]:
            f.write(f"{_RMSE},")
        f.write('\nMAPE,')
        for (_, _, _MAPE, _) in interval_lower_detect_result[:, -1]:
            f.write(f"{_MAPE},")
        f.write('\nME,')
        for (_, _, _, _ME) in interval_lower_detect_result[:, -1]:
            f.write(f"{_ME},")
        
        for interval, __, _, (_MAE, _RMSE, _MAPE, _ME) in interval_lower_detect_result:
            print(interval, _MAE, _RMSE, _MAPE, _ME)
    
    with open('result_data/(BN0)bonduary_detect_result.csv', 'w') as f:
        f.write(f"########,邊界({boundary_upper}、{boundary_lower}),區間檢測,結果,########\n")
        for idx, (low, up, bound_val) in enumerate(interval_upper):
            f.write(f",({idx+1}) boundary±{bound_val}")
        f.write('\nMAE,')
        for idx, ((u_MAE, _, _, _), (l_MAE, _, _, _)) in enumerate(zip(interval_upper_detect_result[:, -1], interval_lower_detect_result[:, -1])):
            if u_MAE == 0 or l_MAE == 0:
                f.write(f"{(u_MAE+l_MAE)},")
            else:
                f.write(f"{(u_MAE+l_MAE)/2.0},")
        f.write('\nRMSE,')
        for idx, ((_, u_RMSE, _, _), (_, l_RMSE, _, _)) in enumerate(zip(interval_upper_detect_result[:, -1], interval_lower_detect_result[:, -1])):
            if u_RMSE == 0 or l_RMSE == 0:
                f.write(f"{(u_RMSE+l_RMSE)},")
            else:
                f.write(f"{(u_RMSE+l_RMSE)/2.0},")
        f.write('\nMAPE,')
        for idx, ((_, _, u_MAPE, _), (_, _, l_MAPE, _)) in enumerate(zip(interval_upper_detect_result[:, -1], interval_lower_detect_result[:, -1])):
            if u_MAPE == 0 or l_MAPE == 0:
                f.write(f"{(u_MAPE+l_MAPE)},")
            else:
                f.write(f"{(u_MAPE+l_MAPE)/2.0},")
        f.write('\nME,')
        for idx, ((_, _, _, u_ME), (_, _, _, l_ME)) in enumerate(zip(interval_upper_detect_result[:, -1], interval_lower_detect_result[:, -1])):
            f.write(f"{np.max((u_ME, l_ME))},")
    
    for idx, data in enumerate(interval_upper_detect_result):
        if (interval_upper_detect_result[idx][-1][0] == 0 or interval_lower_detect_result[idx][-1][0] == 0):
            _MAE = interval_upper_detect_result[idx][-1][0] + interval_lower_detect_result[idx][-1][0]
        else:
            _MAE = (interval_upper_detect_result[idx][-1][0] + interval_lower_detect_result[idx][-1][0]) / 2.0

        if (interval_upper_detect_result[idx][-1][1] == 0 or interval_lower_detect_result[idx][-1][1] == 0):
            _RMSE = interval_upper_detect_result[idx][-1][1] + interval_lower_detect_result[idx][-1][1]
        else:
            _RMSE = (interval_upper_detect_result[idx][-1][1] + interval_lower_detect_result[idx][-1][1]) / 2.0

        if (interval_upper_detect_result[idx][-1][2] == 0 or interval_lower_detect_result[idx][-1][2] == 0):
            _MAPE = interval_upper_detect_result[idx][-1][2] + interval_lower_detect_result[idx][-1][2]
        else:
            _MAPE = (interval_upper_detect_result[idx][-1][2] + interval_lower_detect_result[idx][-1][2]) / 2.0

        _ME = np.max((interval_upper_detect_result[idx][-1][3], interval_lower_detect_result[idx][-1][3]))

        print(f"mean result :: MAE: {_MAE}, RMSE: {_RMSE}, MAPE: {_MAPE}, ME: {_ME}")

    # ------------- RACNN+Classifier 邊界檢測 -------------
    interval_upper_detect_result = []
    for idx1, data in enumerate(interval_upper):
        interval_upper_test_data = []
        interval_upper_pred_data = []
        for idx2, y in enumerate(test_Y_2d):
            if data[0] <= y.numpy() <= data[1]:
                interval_upper_test_data.append(y.numpy())
                interval_upper_pred_data.append(pred_test_net_combine[idx2].numpy())
        if len(interval_upper_test_data) > 0 and len(interval_upper_pred_data) > 0:
            """
            mean_absolute_error = MAE(np.array(interval_upper_test_data), np.array(interval_upper_pred_data))
            root_mean_square_error = RMSE(np.array(interval_upper_test_data), np.array(interval_upper_pred_data))
            mape_val = MAPE(np.array(interval_upper_test_data), np.array(interval_upper_pred_data))
            max_error = np.max(np.abs(np.array(interval_upper_test_data) - np.array(interval_upper_pred_data)))
            """
            mean_absolute_error = MAE(np.array(interval_upper_test_data), np.array(interval_upper_pred_data)).item()
            root_mean_square_error = RMSE(np.array(interval_upper_test_data), np.array(interval_upper_pred_data)).item()
            mape_val = MAPE(np.array(interval_upper_test_data), np.array(interval_upper_pred_data)).item()
            max_error = np.max(np.abs(np.array(interval_upper_test_data) - np.array(interval_upper_pred_data))).item()
        else:
            mean_absolute_error, root_mean_square_error, mape_val, max_error = 0.0, 0.0, 0.0, 0.0
        interval_upper_detect_result.append(
                [np.array(data), 
                np.array(interval_upper_test_data), 
                np.array(interval_upper_pred_data),
                np.array([mean_absolute_error, root_mean_square_error, mape_val, max_error])]
            )
    interval_upper_detect_result = np.array(interval_upper_detect_result, dtype=object)
    # interval_upper_detect_result = np.array(interval_upper_detect_result)

    interval_lower_detect_result = []
    for idx1, data in enumerate(interval_lower):
        interval_lower_test_data = []
        interval_lower_pred_data = []
        for idx2, y in enumerate(test_Y_2d):
            if data[0] <= y.numpy() <= data[1]:
                interval_lower_test_data.append(y.numpy())
                interval_lower_pred_data.append(pred_test_net_combine[idx2].numpy())
        if len(interval_lower_test_data) > 0 and len(interval_lower_pred_data) > 0:
            """
            mean_absolute_error = MAE(np.array(interval_lower_test_data), np.array(interval_lower_pred_data))
            root_mean_square_error = RMSE(np.array(interval_lower_test_data), np.array(interval_lower_pred_data))
            mape_val = MAPE(np.array(interval_lower_test_data), np.array(interval_lower_pred_data))
            max_error = np.max(np.abs(np.array(interval_lower_test_data) - np.array(interval_lower_pred_data)))
            """
            mean_absolute_error = MAE(np.array(interval_lower_test_data), np.array(interval_lower_pred_data)).item()
            root_mean_square_error = RMSE(np.array(interval_lower_test_data), np.array(interval_lower_pred_data)).item()
            mape_val = MAPE(np.array(interval_lower_test_data), np.array(interval_lower_pred_data)).item()
            max_error = np.max(np.abs(np.array(interval_lower_test_data) - np.array(interval_lower_pred_data))).item()
        else:
            mean_absolute_error, root_mean_square_error, mape_val, max_error = 0.0, 0.0, 0.0, 0.0
        interval_lower_detect_result.append(
                [np.array(data), 
                np.array(interval_lower_test_data), 
                np.array(interval_lower_pred_data),
                np.array([mean_absolute_error, root_mean_square_error, mape_val, max_error])]
            )
    interval_lower_detect_result = np.array(interval_lower_detect_result, dtype=object)

    with open('result_data/(RACNN+Classifier)bonduary(upper+lower)_detect_result.csv', 'w') as f:
        f.write(f"########,上邊界({boundary_upper}),區間檢測,結果,########\n")
        for idx, (low, up, bound_val) in enumerate(interval_upper):
            f.write(f",({idx+1}) {boundary_upper}±{bound_val} [{low}~{up}]")
        f.write('\nMAE,')
        for (_MAE, _, _, _) in interval_upper_detect_result[:, -1]:
            f.write(f"{_MAE},")
        f.write('\nRMSE,')
        for (_, _RMSE, _, _) in interval_upper_detect_result[:, -1]:
            f.write(f"{_RMSE},")
        f.write('\nMAPE,')
        for (_, _, _MAPE, _) in interval_upper_detect_result[:, -1]:
            f.write(f"{_MAPE},")
        f.write('\nME,')
        for (_, _, _, _ME) in interval_upper_detect_result[:, -1]:
            f.write(f"{_ME},")
        for interval, __, _, (_MAE, _RMSE, _MAPE, _ME) in interval_upper_detect_result:
            print(interval, _MAE, _RMSE, _MAPE, _ME)
        
        f.write(f"\n\n########,下邊界({boundary_lower}),區間檢測,結果,########\n")
        for idx, (low, up, bound_val) in enumerate(interval_lower):
            f.write(f",({idx+1}) {boundary_lower}±{bound_val} [{low}~{up}]")
        f.write('\nMAE,')
        for (_MAE, _, _, _) in interval_lower_detect_result[:, -1]:
            f.write(f"{_MAE},")
        f.write('\nRMSE,')
        for (_, _RMSE, _, _) in interval_lower_detect_result[:, -1]:
            f.write(f"{_RMSE},")
        f.write('\nMAPE,')
        for (_, _, _MAPE, _) in interval_lower_detect_result[:, -1]:
            f.write(f"{_MAPE},")
        f.write('\nME,')
        for (_, _, _, _ME) in interval_lower_detect_result[:, -1]:
            f.write(f"{_ME},")
    
    with open('result_data/(RACNN+Classifier)bonduary_detect_result.csv', 'w') as f:
        f.write(f"########,邊界({boundary_upper}、{boundary_lower}),區間檢測,結果,########\n")
        for idx, (low, up, bound_val) in enumerate(interval_upper):
            f.write(f",({idx+1}) boundary±{bound_val}")
        f.write('\nMAE,')
        for idx, ((u_MAE, _, _, _), (l_MAE, _, _, _)) in enumerate(zip(interval_upper_detect_result[:, -1], interval_lower_detect_result[:, -1])):
            if u_MAE == 0 or l_MAE == 0:
                f.write(f"{(u_MAE+l_MAE)},")
            else:
                f.write(f"{(u_MAE+l_MAE)/2.0},")
        f.write('\nRMSE,')
        for idx, ((_, u_RMSE, _, _), (_, l_RMSE, _, _)) in enumerate(zip(interval_upper_detect_result[:, -1], interval_lower_detect_result[:, -1])):
            if u_RMSE == 0 or l_RMSE == 0:
                f.write(f"{(u_RMSE+l_RMSE)},")
            else:
                f.write(f"{(u_RMSE+l_RMSE)/2.0},")
        f.write('\nMAPE,')
        for idx, ((_, _, u_MAPE, _), (_, _, l_MAPE, _)) in enumerate(zip(interval_upper_detect_result[:, -1], interval_lower_detect_result[:, -1])):
            if u_MAPE == 0 or l_MAPE == 0:
                f.write(f"{(u_MAPE+l_MAPE)},")
            else:
                f.write(f"{(u_MAPE+l_MAPE)/2.0},")
        f.write('\nME,')
        for idx, ((_, _, _, u_ME), (_, _, _, l_ME)) in enumerate(zip(interval_upper_detect_result[:, -1], interval_lower_detect_result[:, -1])):
            f.write(f"{np.max((u_ME, l_ME))},")
    
    for idx, data in enumerate(interval_upper_detect_result):
        if (interval_upper_detect_result[idx][-1][0] == 0 or interval_lower_detect_result[idx][-1][0] == 0):
            _MAE = interval_upper_detect_result[idx][-1][0] + interval_lower_detect_result[idx][-1][0]
        else:
            _MAE = (interval_upper_detect_result[idx][-1][0] + interval_lower_detect_result[idx][-1][0]) / 2.0

        if (interval_upper_detect_result[idx][-1][1] == 0 or interval_lower_detect_result[idx][-1][1] == 0):
            _RMSE = interval_upper_detect_result[idx][-1][1] + interval_lower_detect_result[idx][-1][1]
        else:
            _RMSE = (interval_upper_detect_result[idx][-1][1] + interval_lower_detect_result[idx][-1][1]) / 2.0

        if (interval_upper_detect_result[idx][-1][2] == 0 or interval_lower_detect_result[idx][-1][2] == 0):
            _MAPE = interval_upper_detect_result[idx][-1][2] + interval_lower_detect_result[idx][-1][2]
        else:
            _MAPE = (interval_upper_detect_result[idx][-1][2] + interval_lower_detect_result[idx][-1][2]) / 2.0

        _ME = np.max((interval_upper_detect_result[idx][-1][3], interval_lower_detect_result[idx][-1][3]))

        print(f"mean result :: MAE: {_MAE}, RMSE: {_RMSE}, MAPE: {_MAPE}, ME: {_ME}")

    # ------------------------- 繪圖 (原始結果與預測) -------------------------
    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    plt.title("HMamba")
    plt.xlabel('Process Data(X)')
    plt.ylabel('CoM')
    plt.plot(test_Y_2d.numpy(), marker=".", linestyle="-", color='blue', label='Ground Truth-CoM')
    plt.plot(mean_pred, marker="o", linestyle="-", color='orange', label='Model Predict-CoM')
    plt.axhline(y=70, color='green', label='Boundary Upper')
    plt.axhline(y=66, color='magenta', linestyle="--", label='Boundary Lower')
    plt.axhline(y=64, color='blue', linestyle="--", label='Sensitivity Upper')
    plt.axhline(y=60, color='red', label='Sensitivity Lower')
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)

    plt.subplot(2, 1, 2)
    plt.title("HMamba")
    plt.xlabel('Process Data(X)')
    plt.ylabel('CoM')
    plt.plot(test_Y_2d.numpy(), marker="*", linestyle="", color='blue', label='Ground Truth-CoM')
    plt.plot(pred_test_net_combine.numpy(), marker="o", linestyle="-", color='orange', label='Model Predict-CoM')
    plt.axhline(y=70, color='green', label='Boundary Upper')
    plt.axhline(y=66, color='magenta', linestyle="--", label='Boundary Lower')
    plt.axhline(y=64, color='blue', linestyle="--", label='Sensitivity Upper')
    plt.axhline(y=60, color='red', label='Sensitivity Lower')
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)
    plt.show()

    # ------------- 熱圖繪製 -------------
    # 這裡直接使用 model 輸出的 att_upper_map 與 att_lower_map
    try:
        attention = [pred_y[3], pred_y[4]]
    except NameError:
        print("請確認 att1 與 att2 注意力圖已定義。")
        attention = [np.zeros((test_Y_2d.shape[0], 1, 9, 9)), np.zeros((test_Y_2d.shape[0], 1, 9, 9))]
    maps = ['RoomTemp','PreformTemp','PreblowPressure','HighPressure','VentPressure',
            'OverallPressure','HighTankPressure','LowTankPressure','HandlePosition']
    result_path = "./result_heatmaps"  # 可根據需求調整
    print("Heatmaps 繪圖開始......")
    heatmap_drawer(attention=attention, gt_value=test_Y_2d.numpy(), pred_value=pred_test_net_combine.numpy(),
                   pred_cls=pred_test_net_cls, maps=maps, result_path=result_path)
    print("Heatmaps 繪圖完成")
