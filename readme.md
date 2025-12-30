![[Figure1_01(1).png]]
# OmniCell 模型使用教程

## 环境准备

### 1. 安装依赖包
```bash
pip install -r requirements.txt
```
### 2. 下载模型检查点
访问 [https://modelscope.cn/models/PJSucas/OmniCell-v1](https://modelscope.cn/models/PJSucas/OmniCell-v1)，下载模型检查点文件到 `OmniCell/checkpoint/` 目录下。

### 3. 下载评测数据
访问 [https://modelscope.cn/datasets/PJSucas/OmniCell-test-data](https://modelscope.cn/datasets/PJSucas/OmniCell-test-data)，下载评测数据集。

## 运行教程

### 启动Jupyter Notebook
在`tutorials/Tutorial_Cluster_blood.ipynb` 文件中学习使用提取细胞embedding的方法
