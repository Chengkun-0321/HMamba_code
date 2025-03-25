import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ============================================ 資料載入 ============================================

# ------------------- 訓練資料 -------------------
CP_data_train_2d_X = np.load('./cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_train_2d_Y = np.load('./cnn-2d_2020-09-09_11-45-24_y.npy')

# 將 NumPy 數據轉換為 PyTorch 張量
CP_data_train_2d_X_tensor = torch.tensor(CP_data_train_2d_X, dtype=torch.float32)
CP_data_train_2d_Y_tensor = torch.tensor(CP_data_train_2d_Y, dtype=torch.float32)

# 假設訓練 X 的形狀為 [3849, 9, 9]，而訓練 Y 的形狀為 [3849, 32]
train_X_2d = CP_data_train_2d_X_tensor  
train_Y_2d = CP_data_train_2d_Y_tensor
print(f"Train X shape: {train_X_2d.shape}")
print(f"Train Y shape: {train_Y_2d.shape}")

# ------------------- 驗證資料 -------------------
CP_data_valid_2d_X = np.load('./validation_data/cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_valid_2d_Y = np.load('./validation_data/cnn-2d_2020-09-09_11-45-24_y.npy')

CP_data_valid_2d_X_tensor = torch.tensor(CP_data_valid_2d_X, dtype=torch.float32)
CP_data_valid_2d_Y_tensor = torch.tensor(CP_data_valid_2d_Y, dtype=torch.float32)

valid_X_2d = CP_data_valid_2d_X_tensor
valid_Y_2d = CP_data_valid_2d_Y_tensor
print(f"Valid X shape: {valid_X_2d.shape}")
print(f"Valid Y shape: {valid_Y_2d.shape}")

# 創建 DataLoader
batch_size = 32
train_dataset = TensorDataset(train_X_2d, train_Y_2d)
valid_dataset = TensorDataset(valid_X_2d, valid_Y_2d)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# ============================================ 模型定義 ============================================

# AdaptiveLoss：根據不同分支輸出計算損失，並根據 Loss 值更新 theta
class AdaptiveLoss(nn.Module):
    def __init__(self, initial_theta=1.0, boundary_upper=70.0, boundary_lower=60.0, center=65.0, min_theta=0.01, eta=0.01):
        super(AdaptiveLoss, self).__init__()
        self.theta = initial_theta
        self.boundary_upper = boundary_upper
        self.boundary_lower = boundary_lower
        self.center = center
        self.min_theta = min_theta
        self.eta = eta

    def loss_mse_test(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def loss_mse_lambda(self, outputs, targets, _lambda):
        total_loss = torch.sum(_lambda * (outputs - targets) ** 2)
        non_zero_lambda = torch.sum(_lambda > 0).item()
        if non_zero_lambda == 0:
            non_zero_lambda = 1
        return total_loss / non_zero_lambda

    def update_theta(self, loss_value):
        self.theta = max(self.min_theta, self.theta - self.eta * loss_value)
        return self.theta

    def forward(self, y_pred, y_true):
        loss_mse = 0
        # 依序對中間、上層、下層計算 MSE loss
        for i in range(len(y_pred)):
            if i == 0:  # 中間層
                l = self.loss_mse_test(y_pred[0], y_true)
                loss_mse += l
            elif i == 1:  # 上層
                lambda_ = y_true - self.center
                lambda_ = torch.where(lambda_ < self.theta, torch.zeros_like(lambda_), lambda_)
                lambda_base = torch.where(lambda_ > self.theta, torch.ones_like(lambda_), lambda_)
                lambda_adjusted = torch.where(
                    y_true >= self.boundary_upper,
                    torch.ones_like(lambda_),
                    torch.tanh(lambda_)
                )
                total_lambda = lambda_base + lambda_adjusted
                l = self.loss_mse_lambda(y_pred[1], y_true, total_lambda)
                loss_mse += l
            elif i == 2:  # 下層
                lambda_ = self.center - y_true
                lambda_ = torch.where(lambda_ < self.theta, torch.zeros_like(lambda_), lambda_)
                lambda_base = torch.where(lambda_ > self.theta, torch.ones_like(lambda_), lambda_)
                lambda_adjusted = torch.where(
                    y_true <= self.boundary_lower,
                    torch.ones_like(lambda_),
                    torch.tanh(lambda_)
                )
                total_lambda = lambda_base + lambda_adjusted
                l = self.loss_mse_lambda(y_pred[2], y_true, total_lambda)
                loss_mse += l
        self.update_theta(loss_mse.item())
        return loss_mse

# MultiHeadSSMBlock：多頭 SSM Block
class MultiHeadSSMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads):
        super(MultiHeadSSMBlock, self).__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 將輸入映射到 hidden_dim
        self.a = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(heads)])
        self.b = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(heads)])
        # 將 hidden_dim 映射到 output_dim；同時直接將原始輸入映射到 output_dim
        self.c = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(heads)])
        self.d = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(heads)])

        # 因果卷積層，每個頭皆獨立
        self.causal_convs = nn.ModuleList([nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=2, dilation=2) for _ in range(heads)])
        self.activation = nn.Tanh()

    def forward(self, x):
        head_outputs = []
        for i in range(self.heads):
            h_t = self.activation(self.a[i](x) + self.b[i](x))  # [batch, seq_len, hidden_dim]
            ssm_output = self.c[i](h_t) + self.d[i](x)            # [batch, seq_len, output_dim]
            ssm_output_trans = ssm_output.transpose(1, 2)         # [batch, output_dim, seq_len]
            conv_output = self.causal_convs[i](ssm_output_trans)    # [batch, output_dim, seq_len]
            conv_output = conv_output.transpose(1, 2)             # [batch, seq_len, output_dim]
            fused_output = ssm_output + conv_output               # 融合全局與局部特徵
            fused_output = self.activation(fused_output)
            head_outputs.append(fused_output)
        # 將多頭結果取平均，輸出形狀仍為 [batch, seq_len, output_dim]
        multihead_output = torch.mean(torch.stack(head_outputs, dim=-1), dim=-1)
        return multihead_output

# Self-Attention 機制
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.scale = embed_dim ** -0.5
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        scores = torch.einsum("bqd,bkd->bqk", Q, K) * self.scale  # [batch, seq_len, seq_len]
        weights = F.softmax(scores, dim=-1)
        attention = torch.einsum("bqk,bvd->bqd", weights, V)
        return self.fc(attention)

# MambaStructuredBackbone：結合 Conv1d 與 SSM 模塊
class MambaStructuredBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads):
        super(MambaStructuredBackbone, self).__init__()
        # conv1 用於初步特徵提取，輸入通道為 input_dim
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        # conv2 的 in_channels 與 ssm 輸出通道相同（output_dim）
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
        self.ssm = MultiHeadSSMBlock(input_dim, hidden_dim, output_dim, heads)

    def forward(self, x):
        # x shape 為 [batch, seq_len, features]
        x_conv = F.relu(self.conv1(x.transpose(1, 2)))  # [batch, hidden_dim, seq_len]
        x_conv = x_conv.transpose(1, 2)                   # [batch, seq_len, hidden_dim]
        x_ssm = self.ssm(x)                             # [batch, seq_len, output_dim]
        x_out = F.relu(self.conv2(x_ssm.transpose(1, 2)))  # [batch, output_dim, seq_len]
        return x_out.transpose(1, 2)                      # [batch, seq_len, output_dim]

# VirtualMetrologySelector：根據中間層結果決定最終輸出採用上層或下層預測
class VirtualMetrologySelector(nn.Module):
    def __init__(self, boundary, delta):
        super(VirtualMetrologySelector, self).__init__()
        self.boundary = boundary
        self.delta = delta

    def forward(self, y_middle, y_upper, y_lower):
        # 當 y_middle 大於 (boundary + delta) 時，選擇 y_upper；否則保留 y_middle
        output = torch.where(y_middle > (self.boundary + self.delta), y_upper, y_middle)
        # 當 y_middle 小於 (boundary - delta) 時，選擇 y_lower；否則保持原值
        output = torch.where(y_middle < (self.boundary - self.delta), y_lower, output)
        return output

# HMambaSA：結合中間層、上層與下層
# ★ 這裡修改 forward，使得每個分支的序列輸出只取最後一個時間步，最終輸出形狀為 [batch, output_dim]
class HMambaSA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads, boundary, delta):
        super(HMambaSA, self).__init__()
        self.middle_layer = MambaStructuredBackbone(input_dim, hidden_dim, output_dim, heads)
        # 注意：這裡讓 SelfAttention 以 input_dim 為 embed_dim（因為上層輸入 x 的特徵數為 input_dim）
        self.upper_layer = nn.Sequential(
            SelfAttention(input_dim),
            MambaStructuredBackbone(input_dim, hidden_dim, output_dim, heads)
        )
        self.lower_layer = nn.Sequential(
            SelfAttention(input_dim),
            MambaStructuredBackbone(input_dim, hidden_dim, output_dim, heads)
        )
        self.selector = VirtualMetrologySelector(boundary, delta)

    def forward(self, x):
        # x shape 為 [batch, seq_len, features]；例如 [32, 9, 9]
        y_middle_seq = self.middle_layer(x)   # [batch, seq_len, output_dim]
        y_upper_seq  = self.upper_layer(x)
        y_lower_seq  = self.lower_layer(x)
        # 將各分支的序列輸出降採樣（例如取最後一個時間步）以匹配目標形狀 [batch, output_dim]
        y_middle = y_middle_seq[:, -1, :]
        y_upper  = y_upper_seq[:, -1, :]
        y_lower  = y_lower_seq[:, -1, :]
        # 如果需要融合最終預測，可透過 selector（這裡返回三個分支，供損失函數計算使用）
        return (y_middle, y_upper, y_lower)

# MAPE 計算函數
def calculate_mape(pred, true, epsilon=1e-8):
    mask = true != 0  # 避免除以零
    return torch.mean(torch.abs((pred[mask] - true[mask]) / (true[mask] + epsilon))) * 100

# ============================================ 訓練流程 ============================================

if __name__ == "__main__":
    # 訓練參數：
    # 這裡 input_dim 與 X 的特徵數均為 9，
    # 將 output_dim 設為 32，以匹配目標 Y 的維度（[batch, 32]），
    # 且由於我們在 HMambaSA 中取了序列的最後一個時間步，
    # 模型最終輸出形狀將為 [batch, 32]。
    input_dim = 9      
    hidden_dim = 20
    output_dim = 32    # 修改為 32 以匹配目標
    heads = 6
    boundary = 65.0
    delta = 0.1
    theta = 1
    boundary_upper = 70.0
    boundary_lower = 60.0
    epochs = 30000
    valid_period = 10
    learning_rate = 0.001

    model = HMambaSA(input_dim, hidden_dim, output_dim, heads, boundary=boundary, delta=delta)
    loss_fn = AdaptiveLoss(initial_theta=theta, boundary_upper=boundary_upper, boundary_lower=boundary_lower, center=boundary)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_data = []
    train_mape_middle, train_mape_upper, train_mape_lower = [], [], []
    valid_mape_middle, valid_mape_upper, valid_mape_lower = [], [], []

    try:
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                # 注意：batch_X shape 為 [batch, seq_len, features]，例如 [32, 9, 9]
                predictions = model(batch_X)  # 每個分支輸出形狀皆為 [batch, output_dim] (即 [32,32])
                loss = loss_fn(predictions, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            loss_data.append(avg_train_loss)
            print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Theta: {loss_fn.theta:.4f}")

            # 每 valid_period 次進行驗證
            if (epoch + 1) % valid_period == 0:
                model.eval()
                with torch.no_grad():
                    train_mape_middle_epoch, train_mape_upper_epoch, train_mape_lower_epoch = 0, 0, 0
                    valid_mape_middle_epoch, valid_mape_upper_epoch, valid_mape_lower_epoch = 0, 0, 0

                    # 訓練集 MAPE 計算
                    for batch_X, batch_Y in train_loader:
                        predictions = model(batch_X)
                        train_mape_middle_epoch += calculate_mape(predictions[0], batch_Y)
                        train_mape_upper_epoch += calculate_mape(predictions[1], batch_Y)
                        train_mape_lower_epoch += calculate_mape(predictions[2], batch_Y)

                    # 驗證集 MAPE 計算
                    for batch_X, batch_Y in valid_loader:
                        predictions = model(batch_X)
                        valid_mape_middle_epoch += calculate_mape(predictions[0], batch_Y)
                        valid_mape_upper_epoch += calculate_mape(predictions[1], batch_Y)
                        valid_mape_lower_epoch += calculate_mape(predictions[2], batch_Y)

                    train_mape_middle.append(train_mape_middle_epoch / len(train_loader))
                    train_mape_upper.append(train_mape_upper_epoch / len(train_loader))
                    train_mape_lower.append(train_mape_lower_epoch / len(train_loader))
                    valid_mape_middle.append(valid_mape_middle_epoch / len(valid_loader))
                    valid_mape_upper.append(valid_mape_upper_epoch / len(valid_loader))
                    valid_mape_lower.append(valid_mape_lower_epoch / len(valid_loader))

                    print(f"Epoch {epoch + 1}:")
                    print(f"Train MAPE - Middle: {train_mape_middle[-1]:.4f}%, Upper: {train_mape_upper[-1]:.4f}%, Lower: {train_mape_lower[-1]:.4f}%")
                    print(f"Valid MAPE - Middle: {valid_mape_middle[-1]:.4f}%, Upper: {valid_mape_upper[-1]:.4f}%, Lower: {valid_mape_lower[-1]:.4f}%")
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    # 儲存訓練結果
    np.save("loss_data.npy", loss_data)
    print("Training completed. Results saved.")
