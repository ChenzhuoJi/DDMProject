
配套 NeurIPS 论文"离散状态空间中的结构化去噪扩散模型"的代码。

```
@inproceedings{austin2021structured,
author    = {Jacob Austin and
             Daniel D. Johnson and
             Jonathan Ho and
             Daniel Tarlow and
             Rianne van den Berg},
title     = {Structured Denoising Diffusion Models in Discrete State-Spaces},
booktitle = {Advances in Neural Information Processing Systems},
year      = {2021}
}
```

`images` 子目录包含图像空间中的 D3PM 代码，可用于复现我们第 6 节的实验。
其中包含评估 bits-per-dimension 指标的代码，但 FID 和 IS 指标的评估代码目前尚未提供。

`text` 子目录包含文本空间中的 D3PM 代码，对应我们第 5 节的实验。

`insertdelete` 子目录包含通过插入和删除操作增强的 D3PM 代码，如后续论文"Beyond In-Place Corruption: Insertion and Deletion In Denoising Probabilistic Models"所述。

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
