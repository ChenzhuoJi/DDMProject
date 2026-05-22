
本目录包含 ICML 2021 INNF+ 研讨会论文"Beyond In-Place Corruption: Insertion and Deletion In Denoising Probabilistic Models"的代码。

```
@inproceedings{johnson2021beyond,
author    = {Daniel D. Johnson and
             Jacob Austin and
             Rianne van den Berg and
             Daniel Tarlow},
title     = {Beyond In-Place Corruption: Insertion and Deletion In Denoising Probabilistic Models},
booktitle = {ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models},
year      = {2021}
}
```

## 交互式引导笔记本

您可以通过以下交互式指南探索插入-删除前向过程：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook_demo]

[notebook_demo]: https://colab.research.google.com/github/google-research/google-research/blob/master/d3pm/insertdelete/Insertion_and_Deletion_Forward_Process_Guide.ipynb

## 代码组织

代码组织如下：

- 主要逻辑
  - `forward_process.py` 包含基于概率有限状态转换器（Probabilistic Finite State Transducers）构建、采样和运行推理的代码。
  - `transition_operator.py` 定义了处理前向过程内部马尔可夫转移矩阵的类。
  - `schedules.py` 包含用于构建扩散调度（决定前向过程混合速度）的辅助类和函数。
  - `training_setup.py` 包含用于构建训练损失和调度的顶层代码。（由于对非开源库的依赖，该代码以黑盒模型预测函数的形式发布，目前不包含构建和训练模型本身的逻辑。）

- 工具模块
  - `distributions.py` 包含前向过程使用的各种相关概率分布的实现。
  - `dynamic_programs.py` 包含用于运行动态规划计算的 JAX 逻辑，可用于执行更昂贵的推理步骤（尽管模型训练并不需要这些）。
  - `math_util.py` 包含各种与数学相关的工具函数。
  - `util.py` 包含其他杂项工具函数。
