
# D3PM 图像实验

本子目录包含 D3PM 图像生成模型的实现。

实现 D3PM 扩散过程的文件：`diffusion-categorical.py`

定义训练循环和实验框架的文件：
- `main.py` 是主可执行文件，实例化所有必要组件。
- `entry_point.py` 负责设置训练任务并读取参数。
- `gm.py` 实现了通用的可训练模型和训练循环。

设置模型和数据的文件：
- `config.py` 构建实验所用的配置对象。
- `model.py` 在 Flax 中实现了 `unet0` 模型。
- `datasets.py` 提供对 CIFAR-10 数据集的访问。

工具模块：
- `utils.py` 定义了各种其他辅助函数。
