# CNN-LSTM Temperature Prediction

[中文](#中文) · [English](#english)

## 中文

用 PyTorch 做气温时间序列预测的三个实验脚本，数据来自 Daily Delhi Climate。模型由一维卷积、LSTM 和线性输出层组成，使用过去 12 个时间步预测下一步。

### 三个脚本

| 文件 | 内容 |
| --- | --- |
| `main.py` | 平均气温单变量预测，绘制原始气温与预测结果 |
| `main2.py` | 单变量预测，增加 MAE、RMSE、R² 等指标和模型保存 |
| `main3.py` | 同时预测平均气温、湿度、风速和气压，逐变量计算指标 |

`main2.py` 和 `main3.py` 默认训练 500 轮，支持有可用 CUDA 时使用 GPU。训练参数直接写在脚本里。

### 运行

```bash
git clone https://github.com/ARETE-zzwl/pytorch-cnn-lstm-temperature-prediction.git
cd pytorch-cnn-lstm-temperature-prediction
python -m pip install -r requirements.txt
python main2.py
```

也可以单独运行 `python main.py` 或 `python main3.py`。请从仓库根目录执行，以便解析 `data/` 的相对路径。图表由 Matplotlib 显示，部分绘图窗口关闭后脚本才会继续。

### 数据和输出

仓库包含 `data/DailyDelhiClimateTrain.csv` 和 `data/DailyDelhiClimateTest.csv`。目前三个脚本都读取 Train 文件，并按时间顺序作 80% / 20% 划分，没有使用单独的 Test 文件。

`main2.py` 和 `main3.py` 会在 `checkpoints/` 保存最佳权重、最后一轮权重和归一化器。两者共用部分输出文件名，连续运行时会覆盖对应文件；需要保留结果时请分别备份。

### 结果怎么理解

这些脚本适合学习序列构建、模型训练和绘图。当前实现先对整份输入数据拟合归一化器，再划分训练与测试，因此存在测试数据参与预处理的问题；`main2.py` 和 `main3.py` 还使用测试段指标挑选最佳权重。终端最后打印的全样本指标也包含训练段，不能当作独立测试结果。

如果要做正式模型比较，需要先划分数据、只用训练段拟合归一化器，并把验证集与最终测试集分开。三个脚本固定随机种子为 42，硬件和 PyTorch 版本仍可能影响结果。

仓库未附代码许可证；数据的使用和再分发也需要确认原始授权。

## English

Three PyTorch experiments for next-step forecasting on the Daily Delhi Climate dataset. Each model combines a 1D convolution, an LSTM and a linear output layer, using a 12-step input window.

### Scripts

- `main.py`: univariate mean-temperature forecasting with original-temperature and prediction plots.
- `main2.py`: univariate forecasting with evaluation metrics and saved weights.
- `main3.py`: joint forecasting of mean temperature, humidity, wind speed and pressure.

`main2.py` and `main3.py` default to 500 epochs and use CUDA when available. Training settings are defined in each script.

### Run

```bash
git clone https://github.com/ARETE-zzwl/pytorch-cnn-lstm-temperature-prediction.git
cd pytorch-cnn-lstm-temperature-prediction
python -m pip install -r requirements.txt
python main2.py
```

Run `python main.py` or `python main3.py` for the other experiments. Use the repository root as the working directory. Matplotlib displays the plots; some windows must be closed before execution continues.

### Data and output

Both Train and Test CSV files are included under `data/`. All three scripts currently read the Train file and split it chronologically into 80% training and 20% testing; they do not read the separate Test file.

`main2.py` and `main3.py` save best and final weights plus scalers under `checkpoints/`. Some filenames are shared, so preserve outputs separately when running both scripts.

### Reading the results

These are learning experiments. Scalers are fitted on the full input before the train/test split, so test data influences preprocessing. `main2.py` and `main3.py` also select the best weights using test-segment metrics. Final full-sample metrics include training data and are not independent test scores.

For a formal comparison, split the data first, fit scalers on training data only, and keep validation separate from the final test. Seeds are set to 42, but results can vary with hardware and PyTorch versions.

No code license is included. Confirm the original dataset's terms before reuse or redistribution.
