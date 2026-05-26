"""
噪声调度模块 — 控制扩散过程中的加噪速率。

定义了多种噪声调度策略：
  - loglinear  : MDLM 默认调度，sigma(t) = -log(1 - t)，move_chance 线性增长
  - cosine     : 余弦调度，在中间区域加噪更慢
  - cosinesqr  : 余弦平方调度
  - linear     : 线性调度 sigma(t) = sigma_min + t * (sigma_max - sigma_min)
  - geometric  : 几何调度
"""

import abc

import torch
import torch.nn as nn

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def get_noise(config, dtype=torch.float32):
  """根据配置选择噪声调度器。"""
  if config.noise.type == 'geometric':
    return GeometricNoise(config.noise.sigma_min,
                          config.noise.sigma_max)
  elif config.noise.type == 'loglinear':
    return LogLinearNoise()
  elif config.noise.type == 'cosine':
    return CosineNoise()
  elif config.noise.type == 'cosinesqr':
    return CosineSqrNoise()
  elif config.noise.type == 'linear':
    return Linear(config.noise.sigma_min,
                  config.noise.sigma_max,
                  dtype)
  else:
    raise ValueError(f'{config.noise.type} is not a valid noise')


def binary_discretization(z):
  """将连续得分离散化为二值信号（用于训练技巧）。"""
  z_hard = torch.sign(z)
  z_soft = z / torch.norm(z, dim=-1, keepdim=True)
  return z_soft + (z_hard - z_soft).detach()


class Noise(abc.ABC, nn.Module):
  """
  噪声调度基类。

  对于时间 t ∈ [0, 1]，定义：
    - total_noise(t) = σ(t)：累积噪声量
    - rate_noise(t)  = g(t)：噪声变化率
  """
  def forward(self, t):
    return self.total_noise(t), self.rate_noise(t)

  @abc.abstractmethod
  def rate_noise(self, t):
    """噪声变化率 g(t)。"""
    pass

  @abc.abstractmethod
  def total_noise(self, t):
    """累积噪声 σ(t) = ∫₀ᵗ g(s) ds + g(0)。"""
    pass


class CosineNoise(Noise):
  """
  余弦噪声调度。

  σ(t) = -log(eps + (1-eps) * cos(πt/2))
  在中间区域加噪较慢，两端较快。
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def rate_noise(self, t):
    cos = (1 - self.eps) * torch.cos(t * torch.pi / 2)
    sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
    scale = torch.pi / 2
    return scale * sin / (cos + self.eps)

  def total_noise(self, t):
    cos = torch.cos(t * torch.pi / 2)
    return - torch.log(self.eps + (1 - self.eps) * cos)


class CosineSqrNoise(Noise):
  """余弦平方噪声调度。"""
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def rate_noise(self, t):
    cos = (1 - self.eps) * (
      torch.cos(t * torch.pi / 2) ** 2)
    sin = (1 - self.eps) * torch.sin(t * torch.pi)
    scale = torch.pi / 2
    return scale * sin / (cos + self.eps)

  def total_noise(self, t):
    cos = torch.cos(t * torch.pi / 2) ** 2
    return - torch.log(self.eps + (1 - self.eps) * cos)


class Linear(Noise):
  """线性噪声调度 σ(t) = σ_min + t * (σ_max - σ_min)。"""
  def __init__(self, sigma_min=0, sigma_max=10, dtype=torch.float32):
    super().__init__()
    self.sigma_min = torch.tensor(sigma_min, dtype=dtype)
    self.sigma_max = torch.tensor(sigma_max, dtype=dtype)

  def rate_noise(self, t):
    return self.sigma_max - self.sigma_min

  def total_noise(self, t):
    return self.sigma_min + t * (self.sigma_max - self.sigma_min)

  def importance_sampling_transformation(self, t):
    """重要性采样变换，用于训练时的时间重采样。"""
    f_T = torch.log1p(- torch.exp(- self.sigma_max))
    f_0 = torch.log1p(- torch.exp(- self.sigma_min))
    sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
    return (sigma_t - self.sigma_min) / (
      self.sigma_max - self.sigma_min)


class GeometricNoise(Noise):
  """几何噪声调度。"""
  def __init__(self, sigma_min=1e-3, sigma_max=1):
    super().__init__()
    self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

  def rate_noise(self, t):
    return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (
      self.sigmas[1].log() - self.sigmas[0].log())

  def total_noise(self, t):
    return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise):
  """
  Log-Linear 噪声调度（MDLM 默认）。

  性质：
    - move_chance(t) = 1 - exp(-σ(t)) ≈ (1-eps) * t
    - σ(t) = -log(1 - (1-eps) * t)
    - 当 t ∈ [0,1] 时，move_chance 从 0 近似线性增长到 ~1

  这是 MDLM 的关键设计：masking 概率随时间线性增长，
  使得每个时间步的噪声量均匀。
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.sigma_max = self.total_noise(torch.tensor(1.0))
    self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

  def rate_noise(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  def total_noise(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def importance_sampling_transformation(self, t):
    """重要性采样变换。"""
    f_T = torch.log1p(- torch.exp(- self.sigma_max))
    f_0 = torch.log1p(- torch.exp(- self.sigma_min))
    sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
    t = - torch.expm1(- sigma_t) / (1 - self.eps)
    return t
