Spring 2026

## 1.1 项目介绍

为主动应对智能时代对高等教育人才培养提出的新挑战，推进人工智能技术与教育教学的深度融合，聚焦课程设计从“内容导向”转为“问题导向”，积极探索“师—机—生”协同共创新范式，提升学生的自主学习能力、创新思维及解决复杂问题的能力，《随机过程》课程开展“问题导向的研讨式学习”教学模式探索项目。

本课程项目要求同学围绕近期兴起的离散扩散(语言)模型，特别是 Masked Diffusion Model，MaskedDiffusion Language Model 范式，完成一次完整的文献调研、方法推导、小规模复现实验的问题导向型探索性研究。

## 1.1.1 问题背景

连续扩散模型在图像、音频和视频生成中取得了显著成功，其基本思想是在前向过程中逐步破坏数据，在反向过程中学习逐步去噪。然而，文本由离散 token 构成，不能像图像像素或 latent vector 一样自然地添加高斯噪声。因此，语言 diffusion 的核心难点是：

如何在离散 token 空间中定义合理的前向噪声过程与反向生成过程？

一类重要的技术路线是 mask diffusion。它把文本 token 逐步替换成特殊符号 mask，然后训练模型从部分可见、部分被 mask 的文本中恢复原始 token。这一思想将 BERT 式 masked language modeling、MaskGIT 式并行生成、以及 diffusion probabilistic modeling 联系起来，是当前 diffusion languagemodel 走向基础模型的核心范式之一。

## 1.1.2 参考文献

• Austin et al., Structured Denoising Diffusion Models in Discrete State-Spaces, NeurIPS 2021.

• Campbell et al., A Continuous Time Framework for Discrete Denoising Models, NeurIPS 2022.

• Zhao et al., Unified Discrete Diffusion for Categorical Data, JMLR 2025.

• Shi et al., Simplified and generalized masked diffusion for discrete data, NeurIPS 2024.

• Sahoo et al., Simple and Effective Masked Diffusion Language Models, NeurIPS 2024.

• Nie et al., Large Language Diffusion Models, NeurIPS 2025.

• Ye et al., Dream 7B: Diffusion Large Language Models, 2025.

## 1.2 作业要求

• 调研基于 CTMC 的 Discrete Diffusion Model相关文献 (Deep Research)；

• 研究目前基于 CTMC 的 Discrete Diffusion Model 的基本原理；

• 研究 Masked Diffusion Model 的基本原理；

• (选做) 复现 Masked Diffusion Model；

• (选做) 探索 ”Large” Diffusion Language Model，例如 LLaDA；

• (选做) 探索更具体的理论研究、算法设计、下游任务...

• 不少于5人组队，形成一篇中文研究报告，可使用AI工具润色，需注明每个人的贡献。

## 1.3 作业评价

• 研究报告占平时作业30分中的10分；

• 评价维度:

– 相关文献的完整性；

– 模型基本原理的详实程度；

– 选作内容的探索程度。

• 遴选2-3组评为优秀小组，每个组员可获得额外5-10分期末总评，并在课上汇报研究成果(每组45分钟)。

## 1.4 时间线

• Stage 1：智慧小雅提交研究报告 5.19 20：00 - 5.29 24：00

• Stage 2：遴选优秀小组 6.2 随堂公布结果

• Stage 3：优秀小组汇报 6.9、6.16