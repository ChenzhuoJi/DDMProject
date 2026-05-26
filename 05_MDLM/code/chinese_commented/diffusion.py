import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  """
  MDLM (Masked Diffusion Language Model) 核心类。

  基于连续时间离散扩散框架，支持三种参数化方式：
    - subs  : 吸收态 (absorbing state) 扩散，对应于 D3PM 的特殊情况
    - sedd  : 基于得分的离散扩散 (Score Entropy Discrete Diffusion)
    - d3pm  : 离散去噪扩散概率模型 (Discrete Denoising Diffusion Probabilistic Model)

  采样器支持三种策略：
    - ddpm       : 标准祖先采样，每步都跑网络前向
    - ddpm_cache : 带缓存的祖先采样，token 不变时跳过前向计算
    - analytic   : 解析采样 (SEDD 风格)
  """
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    """
    SUBS (吸收态) 参数化 — 对应 D3PM 的 absorbing state 特例。

    核心思想：
    - 对 [MASK] 位置：模型预测 p(x0 | xt)，即从噪声恢复原词
    - 对非 MASK 位置：强制 logits 为 one-hot（确定该 token 不变）
    """
    # 将 [MASK] token 的 log 概率设为 -inf（不允许预测为 MASK）
    logits[:, :, self.mask_index] += self.neg_infinity

    # 对 logits 做 softmax 归一化（减去 log-sum-exp）
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # 对已经非 MASK 的位置：除自身外全设 -inf，保持确定性
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    """D3PM 参数化：标准的分类分布归一化。"""
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    """
    SEDD (Score Entropy Discrete Diffusion) 参数化。

    将网络输出转换为对数得分 (log score)：
      log s(x_t, y) = log p_θ(y | x_t) - log(exp(σ) - 1) - log(K-1)
    其中 K 为词表大小，σ 为噪声水平。
    """
    # 数值稳定的 log(exp(σ) - 1) 计算
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # 将输入 token 自身的得分置零（约束条件）
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance):
    """
    前向扩散过程：以概率 move_chance 将 token 替换为 [MASK]。

    这是 MDLM 的"加噪"过程：
      q(xt | x0) = (1 - move_chance) * δ_{x0} + move_chance * δ_{[MASK]}

    Args:
      x: 原始 token 序列，shape (batch_size, seq_len)
      move_chance: 替换为 [MASK] 的概率，shape (batch_size, 1)
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    """初始化先验：全部填充为 [MASK] token。"""
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    """
    DDPM 带缓存的反向更新步（仅在 loglinear 噪声下可用）。

    相比于 _ddpm_update，缓存了 p(x0|xt) 以避免重复前向计算。
    如果 x 没有变化（所有 token 保持相同），则跳过网络调用。

    数学推导：
      后验分布 q(x_s | x_t, x0) ∝ p_θ(x0 | x_t) · (move_chance_t - move_chance_s)
      其中 s = t - dt 为下一步时间。
      对 [MASK] token 额外加上 move_chance_s 的质量。

    Args:
      x: 当前 token 序列
      t: 当前时间 (从 1→0)
      dt: 步长
      p_x0: 缓存的 p(x0|xt)，None 表示需要重新计算

    Returns:
      (p_x0, x_next): 更新后的缓存值和下一步 token 序列
    """
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    # LogLinear 噪声下，move_chance 简化为 t 本身
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      # 没有缓存时，执行网络前向计算 p(x0 | xt)
      p_x0 = self.forward(x, sigma_t).exp()

    assert move_chance_t.ndim == p_x0.ndim
    # 从 p(x0|xt) 采样下一步 x_s
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    # 已确定（非 MASK）的 token 保持不变
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):
    """
    标准 DDPM 反向更新步（无缓存，每步都跑网络）。

    使用噪声调度 sigma(t) 计算转移概率，从 p_θ(x0|xt) 采样。
    """
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    # 已确定的 token 保持不变
    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _ar_sampler(self, bsz, ret_snapshots=False):
    """
    自回归采样器（用于 backbone='ar' 时）。

    从左到右逐个生成 token，配合 Gumbel-max 技巧采样。
    如果 ret_snapshots=True，在预设的进度百分比处记录中间状态。
    """
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))

    snapshots = []
    if ret_snapshots:
      snapshot_pcts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      snap_idx = 0
      snapshot_at = int(snapshot_pcts[0] * num_pred_tokens)

    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y

      if ret_snapshots and i + 1 == snapshot_at:
        padded = x.clone()
        padded[:, i + 2:] = self.mask_index
        snapshots.append((1.0 - (i + 1) / num_pred_tokens, padded.cpu()))
        snap_idx += 1
        if snap_idx < len(snapshot_pcts):
          snapshot_at = int(snapshot_pcts[snap_idx] * num_pred_tokens)

    if ret_snapshots:
      snapshots.append((0.0, x.clone().cpu()))
      return x, snapshots
    return x

  def _check_snapshot(self, t_val, snapshot_t_values, snap_idx):
    """
    判断当前时刻是否应该记录快照。

    当 t 首次降低到某个预设阈值以下时触发记录。
    阈值列表从大到小排列（0.5 → 0.4 → ... → 0.0），
    每个阈值只触发一次。

    Args:
      t_val: 当前时间步的值
      snapshot_t_values: 预设的阈值列表（递减）
      snap_idx: 当前已触发的阈值索引

    Returns:
      是否应该记录快照
    """
    if snap_idx >= len(snapshot_t_values):
      return False
    return t_val <= snapshot_t_values[snap_idx]

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5, ret_snapshots=False):
    """
    扩散模型的核心采样函数。

    过程：
    1. 从全 [MASK] 状态开始（吸收态先验）
    2. 在 [eps, 1] 区间均匀采样 num_steps 个时间步
    3. 每步根据 t 计算噪声水平，用网络预测 p(x0|xt)
    4. 根据采样器策略（ddpm/ddpm_cache/analytic）更新 token
    5. 所有非 MASK 的 token 保持不变（吸收态特性）
    6. 可选：最终去噪步骤移除剩余噪声

    如果 ret_snapshots=True，在预设的 t 阈值处克隆当前状态，
    返回给调用方用于可视化。

    Args:
      num_steps: 采样步数
      eps: 数值稳定小量
      ret_snapshots: 是否返回中间状态快照

    Returns:
      Token ids tensor, 或 (token_ids, snapshots) if ret_snapshots=True
    """
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar':
      if ret_snapshots:
        return self._ar_sampler(batch_size_per_gpu, ret_snapshots=True)
      return self._ar_sampler(batch_size_per_gpu)
    if num_steps is None:
      num_steps = self.config.sampling.steps
    # 初始化：全 [MASK]
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    # 在 [1, eps] 之间均匀采样时间步
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    # ddpm_cache 的缓存
    p_x0_cache = None

    # --- 快照记录初始化 ---
    snapshots = []
    if ret_snapshots:
      # 在低 t 区域加密采样，因为大部分去噪发生在后期
      snapshot_t_values = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.005, 0.0]
      snapshots.append((1.0, x.clone().cpu()))  # 记录初始状态
      snap_idx = 0

    # --- 主采样循环：从 t=1 到 t≈0 ---
    for i in range(num_steps):
      t_val = timesteps[i].item()
      t = t_val * torch.ones(
        x.shape[0], 1, device=self.device)

      # 三种采样策略之一
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # 如果 token 有变化，禁用缓存（下次需要重跑网络）
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)

      # 检查是否需要记录快照
      if ret_snapshots and self._check_snapshot(t_val, snapshot_t_values, snap_idx):
        snapshots.append((t_val, x.clone().cpu()))
        snap_idx += 1

    # --- 最终去噪步骤 ---
    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        # 用 argmax 做最终预测，移除所有不确定性
        x = self.forward(x, unet_conditioning).argmax(dim=-1)

    if ret_snapshots:
      snapshots.append((0.0, x.clone().cpu()))  # 记录最终状态
      return x, snapshots
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5, ret_snapshots=False):
    """
    恢复模型状态并执行采样。

    流程：
    1. 如果有 EMA，保存当前参数并切换到 EMA 均值
    2. 切换到 eval 模式
    3. 调用 _sample() 执行采样
    4. 恢复原始参数和训练模式

    Args:
      num_steps: 采样步数
      eps: 数值稳定小量
      ret_snapshots: 是否返回中间状态快照

    Returns:
      Token ids tensor, 或 (token_ids, snapshots)
    """
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    result = self._sample(num_steps=num_steps, eps=eps, ret_snapshots=ret_snapshots)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return result

  def snapshots_to_text(self, snapshots, max_length=80):
    """
    将快照列表转换为可读文本。

    遍历每个时间步的 token 序列，将连续的非 MASK token 分组解码
    （正确处理 BPE 子词合并），用 [MASK] 标记未揭开的 token。

    例如：token 序列 [MASK, 123, 456, MASK, 789]
          → "[MASK] learning [MASK] ."

    Args:
      snapshots: (t_val, tokens_cpu) 对组成的列表
      max_length: 每行最大字符数

    Returns:
      [(t_val, text), ...] 用于显示或保存
    """
    lines = []
    for t_val, tokens in snapshots:
      ids = tokens[0].tolist()
      segments = []
      buf = []
      for tid in ids:
        if tid == self.mask_index:
          if buf:
            segments.append(self.tokenizer.decode(buf))
            buf = []
          segments.append('[MASK]')
        else:
          buf.append(tid)
      if buf:
        segments.append(self.tokenizer.decode(buf))
      text = ' '.join(s for s in segments if s)
      text = ' '.join(text.split())
      if len(text) > max_length:
        text = text[:max_length] + '...'
      lines.append((t_val, text))
    return lines

  def display_snapshots(self, snapshots, max_length=80):
    """
    打印扩散逆向过程的渐进式可视化。

    输出示例：
      ==============================================================
      扩散逆向过程 (t=1.0 → t=0.0)
      ==============================================================
      t=1.000 (  0% 完成): [MASK] [MASK] [MASK] ...
      t=0.500 ( 50% 完成): music brain [MASK] [MASK] ...
      ...
      t=0.000 (100% 完成): music brain' know wearing ...
      ==============================================================

    Args:
      snapshots: _sample() 返回的快照列表
      max_length: 每行最大字符数
    """
    lines = self.snapshots_to_text(snapshots, max_length)
    print('=' * 70)
    print('扩散逆向过程 (t=1.0 → t=0.0)')
    print('=' * 70)
    for t_val, text in lines:
      pct = (1 - t_val) * 100
      print(f't={t_val:.3f} ({pct:3.0f}% 完成): {text}')
    print('=' * 70)

  def save_snapshots_to_file(self, snapshots, filepath, max_length=200):
    """
    将快照可视化保存到文本文件。

    Args:
      snapshots: _sample() 返回的快照列表
      filepath: 输出文件路径
      max_length: 每行最大字符数
    """
    lines = self.snapshots_to_text(snapshots, max_length)
    with open(filepath, 'w') as f:
      f.write('=' * 70 + '\n')
      f.write('扩散逆向过程 (t=1.0 → t=0.0)\n')
      f.write('=' * 70 + '\n')
      for t_val, text in lines:
        pct = (1 - t_val) * 100
        f.write(f't={t_val:.3f} ({pct:3.0f}% 完成): {text}\n')
      f.write('=' * 70 + '\n')

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0):
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance)
    model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths
