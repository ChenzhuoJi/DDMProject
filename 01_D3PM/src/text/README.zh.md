
# D3PM 文本实验

本子目录包含 D3PM 文本生成模型的实现。

该目录默认支持在 LM1B 和 text8 上进行训练。LM1B 由 TFDS 提供，但 text8 需要手动下载。您可以自行下载并将其解压到 `data/` 目录中（来自 http://mattmahoney.net/dc/text8.zip），或者让数据加载器自动下载。

实现 D3PM 扩散过程的文件：`diffusion.py`

定义训练循环和实验框架的文件：
- `main.py` 是主可执行文件，实例化所有必要组件。
- `trainers.py` 实现了通用的可训练模型和训练循环。

设置模型和数据的文件：
- `configs.py` 构建实验所用的配置对象。
- `model.py` 实现了 D3PM 实验所用的核心 Transformer 模型。
- `datasets.py` 提供对 text8 和 LM1B 数据集的访问。

工具模块：
- `types.py` 定义了一些常用的类型和数据结构。
- `utils.py` 定义了各种其他辅助函数。
