import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from einops import rearrange

# ============================ 超參數與裝置設定 ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 根據資料設定（訓練 X 為 [3849, 9, 9]）
seq_len = 9         # 序列長度
d_model = 9         # 每個時間步的特徵數（與 X 的第二個維度對應）
state_size = 128    # S6 模組內部 state size
output_dim = 32     # 最終模型輸出（對應訓練 Y 的維度）

batch_size = 256
num_epochs = 10

# 是否使用 MAMBA 的分支（本例我們使用它）
USE_MAMBA = True
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = False

# ============================ 模型定義 ============================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: str = 'cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        """
        參數 d_model 為 S6 的輸入 feature 數，本例中在 MambaBlock 中呼叫時傳入 2*d_model (即 18)
        """
        super(S6, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        # A: [d_model, state_size]
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)
        
        # 注意：下面這些變數有部分在不同批次大小下可能會出現不一致，
        # 本例中我們不啟用 DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM，
        # 故這裡在 forward 中會使用 else 分支，直接建立新的 h 張量。
        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        
        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        
        # 當 DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM 啟用時使用（本例未啟用）
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        
    def discretization(self):
        # dB: [batch, seq_len, d_model, state_size]
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        # dA: [batch, seq_len, d_model, state_size]
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        return self.dA, self.dB

    def forward(self, x):
        # x: [batch, seq_len, d_model]，這裡 d_model 為 2*d_model (18) 來自前一層變換
        self.B = self.fc2(x)  # [batch, seq_len, state_size]
        self.C = self.fc3(x)  # [batch, seq_len, state_size]
        self.delta = F.softplus(self.fc1(x))  # [batch, seq_len, d_model]
        
        self.discretization()
        
        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            # 略過此分支（本例不使用）
            raise NotImplementedError("不同的隱藏狀態更新機制未啟用")
        else:
            # 直接建立新的 h 張量
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            # 此處由於 h 為 0，故第一項為 0，故 h 主要來自第二項
            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB
            # y: [batch, seq_len, d_model]
            y = torch.einsum('bln,bldn->bld', self.C, h)
            return y

class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        """
        此處 d_model 為傳入 MambaBlock 時的特徵維度（本例中為輸入 d_model，即 9）
        """
        super(MambaBlock, self).__init__()
        # 內部投影將 d_model 映射至 2*d_model (本例 9 -> 18)
        self.inp_proj = nn.Linear(d_model, 2 * d_model, device=device)
        self.out_proj = nn.Linear(2 * d_model, d_model, device=device)
        # 用於 residual skip 連結（同時將 d_model 映射到 2*d_model）
        self.D = nn.Linear(d_model, 2 * d_model, device=device)
        nn.init.constant_(self.out_proj.bias, 1.0)
        
        # S6 模組，注意傳入的 d_model 此處為 2*d_model (即 18)
        self.S6 = S6(seq_len, 2 * d_model, state_size, device)
        
        # 1D 卷積：我們希望在序列長度方向上做卷積，故先將資料轉置
        # 輸入與輸出 channels 均設定為 2*d_model (18)
        self.conv = nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=3, padding=1, device=device)
        self.conv_linear = nn.Linear(2 * d_model, 2 * d_model, device=device)
        
        self.norm = RMSNorm(d_model, device=device)
        
    def forward(self, x):
        """
        x: [batch, seq_len, d_model] 其中 d_model 為 9
        """
        # RMSNorm
        x_norm = self.norm(x)
        # 投影至 2*d_model
        x_proj = self.inp_proj(x_norm)  # [batch, seq_len, 2*d_model]
        
        # 先轉置以符合 Conv1d 輸入格式: [batch, channels, seq_len]
        x_proj_perm = x_proj.permute(0, 2, 1)
        x_conv = self.conv(x_proj_perm)  # [batch, 2*d_model, seq_len]
        # 再轉回 [batch, seq_len, 2*d_model]
        x_conv = x_conv.permute(0, 2, 1)
        x_conv_act = F.silu(x_conv)
        
        x_conv_out = self.conv_linear(x_conv_act)  # [batch, seq_len, 2*d_model]
        
        # S6 層，輸入 shape 為 [batch, seq_len, 2*d_model]，輸出同樣為 [batch, seq_len, 2*d_model]
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)
        
        # residual skip connection（注意投影 x 至 2*d_model）
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)  # [batch, seq_len, d_model]
        return x_out

class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(Mamba, self).__init__()
        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size, device)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size, device)
        
    def forward(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return x

# 在 Mamba 模型外再加一層，將最終輸出轉換為 target 維度
class MambaModel(nn.Module):
    def __init__(self, seq_len, d_model, state_size, output_dim, device):
        super(MambaModel, self).__init__()
        self.mamba = Mamba(seq_len, d_model, state_size, device)
        # 最終輸出經 flatten 後大小為 seq_len * d_model
        self.final_proj = nn.Linear(seq_len * d_model, output_dim, device=device)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = self.mamba(x)  # [batch, seq_len, d_model]
        x = x.reshape(x.size(0), -1)  # flatten 成 [batch, seq_len * d_model]
        x = self.final_proj(x)  # [batch, output_dim]
        return x

# ============================ 資料載入 ============================

# 訓練資料
CP_data_train_2d_X = np.load('./cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_train_2d_Y = np.load('./cnn-2d_2020-09-09_11-45-24_y.npy')

# 轉換為 torch.Tensor
train_X_2d = torch.tensor(CP_data_train_2d_X, dtype=torch.float32)
train_Y_2d = torch.tensor(CP_data_train_2d_Y, dtype=torch.float32)
print(f"Train X shape: {train_X_2d.shape}")
print(f"Train Y shape: {train_Y_2d.shape}")

# 驗證資料
CP_data_valid_2d_X = np.load('./validation_data/cnn-2d_2020-09-09_11-45-24_x.npy')
CP_data_valid_2d_Y = np.load('./validation_data/cnn-2d_2020-09-09_11-45-24_y.npy')

valid_X_2d = torch.tensor(CP_data_valid_2d_X, dtype=torch.float32)
valid_Y_2d = torch.tensor(CP_data_valid_2d_Y, dtype=torch.float32)
print(f"Valid X shape: {valid_X_2d.shape}")
print(f"Valid Y shape: {valid_Y_2d.shape}")

# 將資料包裝成 TensorDataset
train_dataset = TensorDataset(train_X_2d, train_Y_2d)
valid_dataset = TensorDataset(valid_X_2d, valid_Y_2d)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# ============================ 模型、優化器與損失函數 ============================

model = MambaModel(seq_len, d_model, state_size, output_dim, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ============================ 訓練與驗證函數 ============================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)    # [batch, seq_len, d_model]
        targets = targets.to(device)  # [batch, output_dim]
        optimizer.zero_grad()
        outputs = model(inputs)       # [batch, output_dim]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# ============================ 主訓練流程 ============================

for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    valid_loss = evaluate_model(model, valid_loader, criterion, device)
    print(f"Epoch {epoch}/{num_epochs} --> Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

# 可以選擇儲存模型
torch.save(model.state_dict(), "mamba_model.pth")
print("訓練完成，模型已儲存！")
