#!/usr/bin/env python
# coding: utf-8
# 功能說明：
# 1. 給定.npy資料路徑(X、Y兩種)
# 2. 給定你的X資料是1D還是2D
# 3. 給定訓練/測試資集的比例
# 4. 給定shuffle參數，資料集輸出後的順序是否要打亂

import numpy as np
import datetime

# 加載 CSV 文件
process_data_path = "4869-take-normally-process-order-with1col.csv"
measurement_data_path = "4869-take-normally-metrology-order-with1col-centerofmass.csv"

# 載入 CSV 檔案 (若有標題列，請調整 skiprows 參數)
data_process = np.loadtxt(process_data_path, delimiter=',', skiprows=1)
data_measure = np.loadtxt(measurement_data_path, delimiter=',', skiprows=1)

# 將讀取到的資料存成 .npy 檔案，方便後續讀取
process_npy_path = "4869-take-normally-process-order-with1col.npy"
measurement_npy_path = "4869-take-normally-metrology-order-with1col-centerofmass.npy"
np.save(process_npy_path, data_process)
np.save(measurement_npy_path, data_measure)

class CP_Data_split():
    """
    一次把所有資料打亂，然後一次分割成訓練、驗證、測試三個資料檔案，
    之後可以用 data.Dataset class 讀取檔案。
    """
    # split = 0.8  # 訓練 : 測試 = 8 : 2
    def __init__(self, npy_file_path_X, npy_file_path_Y, shape=1, split=[0.9, 0.1], shuffle=True):
        # train(0.9) →(再分) train(0.9) : valid(0.1)
        # test(0.1)
        train_split = split[0] * 0.9
        valid_split = split[0] * 0.1

        # 載入 .npy 檔案
        x_data_np_ = np.load(npy_file_path_X)
        y_data_np_ = np.load(npy_file_path_Y)  # 讀取所有資料
        r, c = y_data_np_.shape  # r: 行數, c: 欄位數

        # 建立空陣列以儲存排列後的資料
        x_data_np = np.empty_like(x_data_np_)
        y_data_np = np.empty_like(y_data_np_)

        if shuffle is True:
            # 先打亂資料順序
            per = np.random.permutation(r)
            if shape == 1:
                x_data_np = x_data_np_[per, :]
            elif shape == 2:
                x_data_np = x_data_np_[per, :, :]
            y_data_np = y_data_np_[per, :]
        else:
            per = np.array(range(r))
            if shape == 1:
                x_data_np = x_data_np_[per, :]
            elif shape == 2:
                x_data_np = x_data_np_[per, :, :]
            y_data_np = y_data_np_[per, :]

        # 切割訓練資料
        training_data_x = x_data_np[:int(train_split * r)]
        training_data_y = y_data_np[:int(train_split * r)]
        # 切割驗證資料
        start_idx = int(train_split * r)
        end_idx = int(train_split * r) + int(valid_split * r)
        validation_data_x = x_data_np[start_idx:end_idx]
        validation_data_y = y_data_np[start_idx:end_idx]
        # 切割測試資料
        testing_data_x = x_data_np[int(split[0] * r):]
        testing_data_y = y_data_np[int(split[0] * r):]

        # 儲存各資料集的形狀資訊
        self.training_shape_x = training_data_x.shape
        self.validation_shape_x = validation_data_x.shape
        self.testing_shape_x = testing_data_x.shape
        self.training_shape_y = training_data_y.shape
        self.validation_shape_y = validation_data_y.shape
        self.testing_shape_y = testing_data_y.shape

        # 依據 shape 決定檔名
        str_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if shape == 1:
            name = 'cnn-1d_' + str(str_time)
        elif shape == 2:
            name = 'cnn-2d_' + str(str_time)
        else:
            name = 'normal_' + str(str_time)

        base_path = './data'
        self.training_path = base_path + '/training_data/' + name
        self.validation_path = base_path + '/validation_data/' + name
        self.testing_path = base_path + '/testing_data/' + name

        # 儲存資料檔案
        np.save(self.training_path + '_x.npy', training_data_x)
        np.save(self.training_path + '_y.npy', training_data_y)
        np.save(self.validation_path + '_x.npy', validation_data_x)
        np.save(self.validation_path + '_y.npy', validation_data_y)
        np.save(self.testing_path + '_x.npy', testing_data_x)
        np.save(self.testing_path + '_y.npy', testing_data_y)
        print("訓練、驗證、測試資料已經儲存完成。")
        print("\t可以使用 get_file_path() 查看資料路徑")
        print("\t可以使用 get_data_shape() 查看資料狀態")

    def get_file_path(self):
        print("資料路徑如下：")
        print(f"\t{self.training_path}_x.npy")
        print(f"\t{self.training_path}_y.npy")
        print("\t" + "*" * 30)
        print(f"\t{self.validation_path}_x.npy")
        print(f"\t{self.validation_path}_y.npy")
        print("\t" + "*" * 30)
        print(f"\t{self.testing_path}_x.npy")
        print(f"\t{self.testing_path}_y.npy")

    def get_data_shape(self):
        print("資料狀態如下：")
        print(f"\ttraining_shape_x: {self.training_shape_x}")
        print(f"\ttraining_shape_y: {self.training_shape_y}")
        print("\t" + "*" * 30)
        print(f"\tvalidation_shape_x: {self.validation_shape_x}")
        print(f"\tvalidation_shape_y: {self.validation_shape_y}")
        print("\t" + "*" * 30)
        print(f"\ttesting_shape_x: {self.testing_shape_x}")
        print(f"\ttesting_shape_y: {self.testing_shape_y}")

if __name__ == '__main__':
    # 建立資料分割物件並執行分割動作
    splitter = CP_Data_split(process_npy_path, measurement_npy_path, shape=1, split=[0.9, 0.1], shuffle=True)
    splitter.get_file_path()
    splitter.get_data_shape()


#TFT
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 加載 CSV 文件
process_data_path = "Process_1100.csv"
measurement_data_path = "y_test.csv"

# 讀取數據，避免 pandas 自動解析數據類型
process_data = pd.read_csv(process_data_path, header=0, dtype=str, low_memory=False)
measurement_data = pd.read_csv(measurement_data_path, header=None)

# 只保留包含 "FIELD" 的列
process_data = process_data[[col for col in process_data.columns if "FIELD" in col]]

process_data = process_data.apply(pd.to_numeric, errors='coerce').fillna(0)

# 確保 Process_1100.csv 行數足夠（應至少有 1100 * 400 行）
num_samples = measurement_data.shape[0]  # 1100 筆
rows_per_sample = 400
expected_rows = num_samples * rows_per_sample

# 如果 Process_1100.csv 行數超過要求，則裁剪
if process_data.shape[0] > expected_rows:
    process_data = process_data.iloc[:expected_rows, :]

# 降維：每 400 行取均值，縮減成 1100 行
process_data_reduced = process_data.groupby(process_data.index // rows_per_sample).mean()

# 確保降維後數據與 y-SWA.csv 對齊
assert process_data_reduced.shape[0] == measurement_data.shape[0], \
    f"降維後行數 {process_data_reduced.shape[0]} ≠ y-SWA.csv 行數 {measurement_data.shape[0]}"

# 進行數據切割（打亂數據以確保隨機性）
train_x, temp_x, train_y, temp_y = train_test_split(
    process_data_reduced, measurement_data, train_size=1000, random_state=42
)
val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

# 轉換為 float32 類型
train_x = train_x.astype(np.float32).values
train_y = train_y.astype(np.float32).values
val_x = val_x.astype(np.float32).values
val_y = val_y.astype(np.float32).values
test_x = test_x.astype(np.float32).values
test_y = test_y.astype(np.float32).values

# 保存為 CSV 文件
pd.DataFrame(train_x).to_csv("train_x.csv", index=False, header=False)
pd.DataFrame(train_y).to_csv("train_y.csv", index=False, header=False)
pd.DataFrame(val_x).to_csv("val_x.csv", index=False, header=False)
pd.DataFrame(val_y).to_csv("val_y.csv", index=False, header=False)
pd.DataFrame(test_x).to_csv("test_x.csv", index=False, header=False)
pd.DataFrame(test_y).to_csv("test_y.csv", index=False, header=False)

# 保存為 .npy 文件
np.save("train_x.npy", train_x)
np.save("train_y.npy", train_y)
np.save("val_x.npy", val_x)
np.save("val_y.npy", val_y)
np.save("test_x.npy", test_x)
np.save("test_y.npy", test_y)

print("數據處理完成，已保存為 .csv 和 .npy 文件！")
'''