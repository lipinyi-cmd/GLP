# 地下水位时间序列建模

MATLAB 实现的 LSTM、多头自注意力 LSTM（MHSA-LSTM）和支持向量回归（SVM）。三个模型均使用贝叶斯超参数优化，目标函数为四轮扩展窗口时间验证的平均 RMSE。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `LSTM.m` | LSTM 训练与测试入口 |
| `MHSA_LSTM.m` | 多头自注意力 LSTM 训练与测试入口 |
| `svm.m` | 支持向量回归训练与测试入口 |
| `expand_T1_trainAndValidate_trainnet_w_K_Fold_times.m` | LSTM 时间验证目标函数 |
| `expand_T1_att_trainAndValidate_trainnet_w_K_Fold_times.m` | MHSA-LSTM 时间验证目标函数 |
| `expand_kfoldLossSVM_time.m` | SVM 时间验证目标函数 |
| `figure6.m` | LSTM/MHSA-LSTM 的原始水位指标和图表；先运行对应模型 |
| `fig3.m` | SVM 的原始水位指标和图表；先运行 `svm.m` |
| `data/` | 本地输入数据目录 |

## 环境要求

- MATLAB：需支持 `trainnet`、`dlarray`；注意力模型还使用 `selfAttentionLayer` 和 `indexing1dLayer`。
- Deep Learning Toolbox：用于两个 LSTM 模型。
- Statistics and Machine Learning Toolbox：用于 `bayesopt`、`optimizableVariable` 和 `fitrsvm`。
- Econometrics Toolbox：SVM 使用 `lagmatrix` 构造滞后特征。
- Parallel Computing Toolbox：可选；入口的 `useParallel` 默认为 `false`，启用并行优化时需要该工具箱。

## 数据准备

将工作簿放到 `data/X_y_2.xlsx`。脚本按自身所在位置定位文件，也可以修改入口顶部的 `dataFile` 指向自己的数据。

| 工作表 | 格式 |
| --- | --- |
| 首个工作表 / `X` | 外生特征；前六列为数值。两个 LSTM 读取首个工作表，SVM 按名称读取 `X`，因此应将 `X` 放在首位。 |
| `y_30860019` | LSTM 默认目标；第二列为水位。 |
| `y_30860016` | MHSA-LSTM 默认目标；第二列为水位。 |
| `y_30860017` | SVM 默认目标；第二列为水位。 |

各表需按相同时间顺序逐行对应，首行为表头。特征第 1–5 列依次表示辐射、气温、相对湿度、降水和 ET，第六列请根据自己的数据定义；两个 LSTM 使用全部六列并追加前一期水位。SVM 使用降水、ET，以及水位、辐射和湿度的指定滞后特征。

请提供足够的数值行，预先处理缺失值和常量列，避免归一化出现无效数值。原始观测工作簿不随项目发布。

## 运行方法

在 MATLAB 中将当前文件夹切换到本目录，运行一个模型。需要原始水位尺度的指标和附加图表时，接着运行对应的绘图脚本：

```matlab
LSTM
figure6
```

```matlab
MHSA_LSTM
figure6
```

```matlab
svm
fig3
```

绘图脚本读取前一个模型留在工作区的变量，需在同一工作区运行。

## 默认实验设置

| 配置 | LSTM | MHSA-LSTM | SVM |
| --- | --- | --- | --- |
| 目标工作表 | `y_30860019` | `y_30860016` | `y_30860017` |
| 读取行数 | 730 | 2191 | 2191 |
| 时间窗口 | 7 | 7 | 显式滞后，最大 7 期 |
| 初始训练集比例 | 80% | 80% | 80% |
| 最终使用的训练范围 | 全部训练窗口 | 全部训练窗口 | 训练分区末尾 730 个样本 |
| 贝叶斯优化评估次数 | 30 | 30 | 30 |
| 每次网络训练轮数 | 100 | 100 | 不适用 |

站点、数据行数、并行开关可在各入口顶部修改；窗口长度和其他模型参数在相应脚本中修改。保留原有不同站点和样本长度，默认结果不能直接作为同一数据条件下的模型比较。



## 上传 GitHub

将本目录作为仓库根目录。提交 `.m` 源文件、README、`data/README.md` 和 Git 配置文件。`.gitignore` 已排除观测数据、MATLAB 工作区/模型、图窗、输出目录及临时文件；数据保留在本机。

首次初始化可在本目录运行：

```sh
git init
git add .
git diff --cached --stat
git commit -m "Prepare groundwater forecasting code for GitHub"
git branch -M main
```

之后关联自己的 GitHub 仓库并推送。当前项目未指定开源许可证；公开分享或允许他人复用前，由代码权利人选择适合的许可证。
