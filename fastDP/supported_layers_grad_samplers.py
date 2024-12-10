"""Util functions for drawing discrete Gaussian samples.

The following functions implement a vectorized TF version of the sampling
algorithm described in the paper:

The Discrete Gaussian for Differential Privacy
https://arxiv.org/pdf/2004.00010.pdf

Note that the exact sampling implementation should use integer and fractional
parameters only. Here, for experimental purposes, we relax this constraint a bit
and use vectorized implementations of Bernoulli and discrete Laplace sampling
that can take float parameters.
"""

import tensorflow as tf
import tensorflow_probability as tf_prob
import numpy as np


def _sample_discrete_laplace(t, shape):
  """Sample from discrete Laplace with scale t.

  This method is based on the observation that sampling from Z ~ Lap(t) is
  equivalent to sampling X, Y independently from Geo(1 - exp(-1/t)) and take
  Z = X - Y.

  Note also that tensorflow_probability's geometric sampler is based on floating
  operations and may possibly be inexact.

  Args:
    t: The scale of the discrete Laplace distribution.
    shape: The tensor shape of the tensors drawn.

  Returns:
    A tensor of the specified shape filled with random values.
  """
  geometric_probs = 1.0 - tf.exp(-1.0 / tf.cast(t, tf.float64))
  geo1 = tf_prob.distributions.Geometric(probs=geometric_probs).sample(shape)
  geo2 = tf_prob.distributions.Geometric(probs=geometric_probs).sample(shape)
  return tf.cast(geo1 - geo2, tf.int64)


def _sample_bernoulli(p):
  """Sample from Bernoulli(p)."""
  return tf_prob.distributions.Bernoulli(probs=p, dtype=tf.int64).sample()


def _check_input_args(scale, shape, dtype):
  """Checks the input args to the discrete Gaussian sampler."""
  if tf.as_dtype(dtype) not in (tf.int32, tf.int64):
    raise ValueError(
        f'Only tf.int32 and tf.int64 are supported. Found dtype `{dtype}`.')

  checks = [
      tf.compat.v1.assert_non_negative(scale),
      tf.compat.v1.assert_integer(scale)
  ]
  with tf.control_dependencies(checks):
    return tf.identity(scale), shape, dtype


@tf.function
def _sample_discrete_gaussian_helper(scale, shape, dtype):
  """Draw samples from discrete Gaussian, assuming scale >= 0."""
  scale = tf.cast(scale, tf.int64)
  sq_scale = tf.square(scale)

  # Scale for discrete Laplace. The sampling algorithm should be correct
  # for any discrete Laplace scale, and the original paper uses
  # `dlap_scale = floor(scale) + 1`. Here we use `dlap_scale = scale` (where
  # input `scale` is restricted to integers >= 1) to simplify the fraction
  # below. It turns out that for integer scales >= 1, `dlap_scale = scale` gives
  # a good minimum success rate of ~70%, allowing a small oversampling factor.
  dlap_scale = scale
  oversample_factor = 1.5

  # Draw at least some samples in case we got unlucky with small input shape.
  min_n = 1000
  target_n = tf.reduce_prod(tf.cast(shape, tf.int64))
  oversample_n = oversample_factor * tf.cast(target_n, tf.float32)
  draw_n = tf.maximum(min_n, tf.cast(oversample_n, tf.int32))

  accepted_n = tf.constant(0, dtype=target_n.dtype)
  result = tf.zeros((0,), dtype=tf.int64)

  while accepted_n < target_n:
    # Since the number of samples could be different in every retry, we need to
    # manually specify the shape info for TF.
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(result, tf.TensorShape([None]))])
    # Draw samples.
    samples = _sample_discrete_laplace(dlap_scale, shape=(draw_n,))
    z_numer = tf.pow((tf.abs(samples) - scale), 2)
    z_denom = 2 * sq_scale
    bern_probs = tf.exp(-tf.divide(z_numer, z_denom))
    accept = _sample_bernoulli(bern_probs)
    # Keep successful samples and increment counter.
    accepted_samples = samples[tf.equal(accept, 1)]
    accepted_n += tf.size(accepted_samples, out_type=accepted_n.dtype)
    result = tf.concat([result, accepted_samples], axis=0)
    # Reduce the number of draws for any retries.
    draw_n = tf.cast(target_n - accepted_n, tf.float32) * oversample_factor
    draw_n = tf.maximum(min_n, tf.cast(draw_n, tf.int32))

  return tf.cast(tf.reshape(result[:target_n], shape), dtype)


def sample_discrete_gaussian(scale, shape, dtype=tf.int32):
  """Draws (possibly inexact) samples from the discrete Gaussian distribution.

  We relax some integer constraints to use vectorized implementations of
  Bernoulli and discrete Laplace sampling. Integer operations are done in
  tf.int64 as TF does not have direct support for fractions.

  Args:
    scale: The scale of the discrete Gaussian distribution.
    shape: The shape of the output tensor.
    dtype: The type of the output.

  Returns:
    A tensor of the specified shape filled with random values.
  """
  scale, shape, dtype = _check_input_args(scale, shape, dtype)
  return tf.cond(
      tf.equal(scale, 0), lambda: tf.zeros(shape, dtype),
      lambda: _sample_discrete_gaussian_helper(scale, shape, dtype))

# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""
This module is a collection of grad samplers 
- methods to calculate per sample grad norms or gradients
for a layer given 1) inputs, AND/OR 2) grad_outputs.

Supports BK (book-keeping) introduced in 
Bu et al. (arXiv:2210.00038)
Differentially Private Optimization on Large Model at Small Cost

and BiTFiT (bias-term fine-tuning) introduced in
Bu et al. (aarXiv:2210.00036)
Differentially Private Bias-Term only Fine-tuning of Foundation Models

Highlights: this code uses the important "mixed ghost norm" trick to achieve its high efficiency,
adapted and improved from 'Scalable and Efficient Training of Large Convolutional Neural Networks with Differential Privacy'
by Bu et al. See their Section 4.

A large portion of this code is adapted Opacus v0.15 (https://github.com/pytorch/opacus), 
from Private-transformers v0.2.3 (https://github.com/lxuechen/private-transformers),
and from Private-vision v0.1.0 (https://github.com/woodyx218/private_vision)
"""

import torch
import transformers.pytorch_utils
from torch import nn
from torch.functional import F
from transformers.models.t5.modeling_t5 import T5LayerNorm
import config
import time

def mixed_ghost_norm(layer,A,B,conv=False):
    # for linear layers, A is activation, B is backprops;
    # for conv layers, A is unfolded activation, B is inverse folded (flattened) backprops;
    if not hasattr(layer, "use_gc"): # use ghost clipping or not
        if conv==False:
            T = torch.prod(torch.Tensor(list(A.shape[1:-1]))).item()
            #assert T == torch.prod(torch.Tensor(list(B.shape[1:-1]))).item()
            d = A.shape[-1]
            p = B.shape[-1]
        else:
            T = A.shape[-1]
            #assert T == B.shape[-1]
            d = A.shape[1]
            p = B.shape[1]
        d_times_p = torch.prod(torch.Tensor(list(layer.weight.size())))
        layer.use_gc = bool(2*T**2 <= d_times_p)
        #assert d*p == d_times_p
        #print(layer,'\n use ghost clip: ',layer.use_gc,'\n T= ',T,';d= ',d,';p= ',p,';2T^2= ',2*T**2,';pd= ',p*d)

def sum_over_all_but_batch_and_last_n(tensor: torch.Tensor, n_dims: int) -> torch.Tensor:
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)


def _light_linear_weight_norm_sample(A, B) -> torch.Tensor:
    """Compute gradient sample norm for the weight matrix in a linear layer."""
    if A.dim() == 2 and B.dim() == 2:
        return _light_linear_weight_norm_sample_non_sequential(A, B)
    elif A.dim() == 3 and B.dim() == 3:
        return _light_linear_weight_norm_sample_sequential(A, B)
    else:
        raise ValueError(f"Unexpected input shape: {A.size()}, grad_output shape: {B.size()}")


@torch.jit.script
def _light_linear_weight_norm_sample_sequential(A, B):
    """Lightweight norm computation in ghost clipping.

    Linear algebra identity trick -- Eq. 3 in the paper.
    """
    #return torch.sqrt((torch.einsum('bTd,bSd->bTS',A,A)*torch.einsum('bTp,bSp->bTS',B,B)).sum(dim=(1, 2)))
    return torch.sqrt((torch.bmm(A, A.transpose(-1, -2)) * torch.bmm(B, B.transpose(-1, -2))).sum(dim=(1, 2)))


@torch.jit.script
def _light_linear_weight_norm_sample_non_sequential(A, B):
    """The Goodfellow trick, i.e., Frobenius norm equal to product of 2-norms."""
    return A.norm(2, dim=1) * B.norm(2, dim=1)

@torch.jit.script
def _light_linear_bias_norm_sample(B):
    if B.dim() == 2:
        return B.norm(2, dim=1)
    elif B.dim() == 3:
        return B.sum(dim=1).norm(2, dim=1)
    else:
        raise ValueError(f"Unexpected grad_output shape: {B.size()}")

def _compute_linear_grad_sample(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, clipping_mode: str) -> None:
    """Computes per sample gradients for `nn.Linear` layer.
    A is activations or layer's input, see autograd_grad_sample line 229; B is output gradient
    This function is written in an unusually bespoke way to avoid using `torch.einsum`.
    """
    if A!=None:
        if clipping_mode in ['MixGhostClip','MixOpt']:
            mixed_ghost_norm(layer, A, B)
        else:
            layer.use_gc=True
        
        
        if A.dim()>3:
            A=torch.flatten(A,start_dim=1,end_dim=-2)
            B=torch.flatten(B,start_dim=1,end_dim=-2)
            
        if layer.use_gc==True:
            #--- compute weight gradient norm
            layer.weight.norm_sample = _light_linear_weight_norm_sample(A, B)
        else:
            ## Or use Line 105 (v0.1.0) https://github.com/lxuechen/private-transformers/blob/main/private_transformers/privacy_utils/supported_layers_grad_samplers.py
            layer.weight.grad_sample = torch.einsum('b...d, b...p-> bpd', A, B).detach()
            layer.weight.norm_sample = torch.sqrt(torch.sum(layer.weight.grad_sample**2, dim=(1, 2)))
            if clipping_mode!='MixOpt':
                del layer.weight.grad_sample
    
    #--- compute bias gradient norm
    if layer.bias is not None:
        layer.bias.norm_sample = _light_linear_bias_norm_sample(B)
        if B.dim() == 3:
            grad_bias = B.sum(dim=1)
        elif B.dim() == 2:
            grad_bias = B
        layer.bias.grad_sample = grad_bias.detach()     


def _compute_Conv1D_grad_sample(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, clipping_mode: str) -> None:
    """Computes per sample gradients for `nn.Linear` layer.
    A is activations or layer's input, see autograd_grad_sample line 229; B is output gradient
    This function is written in an unusually bespoke way to avoid using `torch.einsum`.
    """
    if A!=None:
        if clipping_mode in ['MixGhostClip','MixOpt']:
            mixed_ghost_norm(layer, A, B)
        else:
            layer.use_gc=True
        
        
        if A.dim()>3:
            A=torch.flatten(A,start_dim=1,end_dim=-2)
            B=torch.flatten(B,start_dim=1,end_dim=-2)
            
        if layer.use_gc==True:
            #--- compute weight gradient norm
            layer.weight.norm_sample = _light_linear_weight_norm_sample(A, B)
        else:
            ## Or use Line 105 (v0.1.0) https://github.com/lxuechen/private-transformers/blob/main/private_transformers/privacy_utils/supported_layers_grad_samplers.py
            layer.weight.grad_sample = torch.einsum('b...d, b...p-> bdp', A, B).detach()
            layer.weight.norm_sample = torch.sqrt(torch.sum(layer.weight.grad_sample**2, dim=(1, 2)))
            if clipping_mode!='MixOpt':
                del layer.weight.grad_sample
    
    #--- compute bias gradient norm
    if layer.bias is not None:
        layer.bias.norm_sample = _light_linear_bias_norm_sample(B)
        if B.dim() == 3:
            grad_bias = B.sum(dim=1)
        elif B.dim() == 2:
            grad_bias = B
        layer.bias.grad_sample = grad_bias.detach()   
        
def _compute_layer_norm_grad_sample(
    layer: nn.LayerNorm,
    A: torch.Tensor, B: torch.Tensor, 
    clipping_mode: str) -> None:
    """Computes per sample gradients for normalization layers."""
    if A!=None:
        #--- weight, compute gradient norm
        grad_sample = sum_over_all_but_batch_and_last_n(
            F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
            layer.weight.dim(),
        )
        norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
        layer.weight.norm_sample = norm_sample
        layer.weight.grad_sample = grad_sample.detach()
    
    #--- bias, compute gradient norm
    if layer.bias is not None:
        grad_sample = sum_over_all_but_batch_and_last_n(B, layer.bias.dim())        
        layer.bias.norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
        layer.bias.grad_sample = grad_sample.detach()

      
def _compute_group_norm_grad_sample(
    layer: nn.GroupNorm,
    A: torch.Tensor, B: torch.Tensor, 
    clipping_mode: str
) -> None:
    
    """Computes per sample gradients for normalization layers."""
    if A!=None:
        grad_sample = torch.einsum('ni...->ni',F.group_norm(A, layer.num_groups, eps=layer.eps) * B)
    
        layer.weight.norm_sample = grad_sample.norm(2, dim=1)
        layer.weight.grad_sample = grad_sample.detach()

    if layer.bias is not None:
        grad_sample = torch.einsum('ni...->ni', B)
        layer.bias.norm_sample = grad_sample.norm(2, dim=1)
        layer.bias.grad_sample = grad_sample.detach()

def _compute_instance_norm_grad_sample(
    layer: nn.InstanceNorm2d,
    A: torch.Tensor, B: torch.Tensor, 
    clipping_mode: str
) -> None:
    
    """Computes per sample gradients for normalization layers."""
    if A!=None:
        grad_sample = torch.einsum('ni...->ni',F.instance_norm(A, eps=layer.eps) * B)
    
        layer.weight.norm_sample = grad_sample.norm(2, dim=1)
        layer.weight.grad_sample = grad_sample.detach()

    if layer.bias is not None:
        grad_sample = torch.einsum('ni...->ni', B)
        layer.bias.norm_sample = grad_sample.norm(2, dim=1)
        layer.bias.grad_sample = grad_sample.detach()

def _compute_embedding_grad_sample(layer: nn.Embedding, A: torch.Tensor, B: torch.Tensor, clipping_mode: str) -> None:
    """Computes per sample gradients for `nn.Embedding` layer."""

    #--- compute gradient norm
    not_AAt: torch.Tensor = ~A[:, :, None].eq(A[:, None, :])
    # Clear the contribution to the norm of the gradient for the padding token.
    #   In vanilla backpropagation, this particular embedding doesn't contribute to the gradient anyway.
    #   For more see 1.10.0 doc: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    #       'the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”.'
    padding_idx = layer.padding_idx
    if padding_idx is not None:
        # The right way to think about the next line of code is that A_i[t, padding_idx] = 0 for all t in [T].
        #   So the entry gets cleared whenever one of A, A^t takes the padding idx.
        not_AAt.bitwise_or_((A[:, :, None] == padding_idx) | (A[:, None, :] == padding_idx))
    norm_sample = torch.sqrt((torch.bmm(B, B.transpose(-1, -2)).masked_fill(not_AAt, 0)).sum(dim=(1, 2)))
    layer.weight.norm_sample = norm_sample


def _compute_conv_grad_sample(layer, A: torch.Tensor, B: torch.Tensor, clipping_mode: str):
    B = B.flatten(2)                                  # F^{-1}(dL/ds)
    # check also https://github.com/woodyx218/private_vision/blob/main/private_vision/privacy_utils/supported_layers_grad_samplers.py
    if A!=None:
        if layer.__class__.__name__=='Conv1d':
            padding = layer.padding if isinstance(
                    layer.padding, tuple) else (*layer.padding, *layer.padding)
            # padded_A = F.pad(A, padding)
            A = F.unfold(A.unsqueeze(-2), kernel_size=(1, *layer.kernel_size),
                                padding=(0, *padding),
                                dilation=(1, *layer.dilation),
                                stride=(1, *layer.stride))
        elif layer.__class__.__name__=='Conv2d':
            A = F.unfold(A, kernel_size=layer.kernel_size,
                                    dilation=layer.dilation, padding=layer.padding,
                                    stride=layer.stride) # U(a)  
        elif layer.__class__.__name__=='Conv3d':
            from opacus.utils import tensor_utils
            A = tensor_utils.unfold3d(A, kernel_size=layer.kernel_size,
                                             dilation=layer.dilation, padding=layer.padding,
                                             stride=layer.stride)
    
        if clipping_mode in ['MixGhostClip','MixOpt']:
            mixed_ghost_norm(layer, A, B,conv=True)
        else:
            layer.use_gc=True
        
        if layer.use_gc==True:
            #--- compute weight gradient norm
            aTa = torch.einsum('bji, bjk -> bik', A, A)
            gTg = torch.einsum('bji, bjk -> bik', B, B)
            #norm_sample = torch.sqrt(torch.einsum('bij, bij -> b', aTa, gTg))
            norm_sample = torch.sqrt((aTa*gTg).sum(dim=(1,2)))    
            layer.weight.norm_sample = norm_sample
        else:
            ## Or use Line 105 https://github.com/lxuechen/private-transformers/blob/main/private_transformers/privacy_utils/supported_layers_grad_samplers.py
            layer.weight.grad_sample = torch.einsum('bd..., bp...-> bpd', A, B).detach()
            layer.weight.norm_sample = torch.sqrt((layer.weight.grad_sample**2).sum(dim=(1, 2)))
            if clipping_mode !='MixOpt':
                del layer.weight.grad_sample

    #--- bias, compute gradient norm
    if layer.bias is not None:
        grad_sample = B.sum(dim=2).detach()
        layer.bias.norm_sample = grad_sample.norm(2, dim=1)
        layer.bias.grad_sample = grad_sample

def _compute_t5_layer_norm_grad_sample(layer: T5LayerNorm, A: torch.Tensor, B: torch.Tensor, clipping_mode: str):
    # `transformers.models.t5.modeling_t5.T5LayerNorm` has single input and output. Unpack singleton tuples.
    # https://github.com/huggingface/transformers/blob/ccc089780415445768bcfd3ac4418cec20353484/src/transformers/models/t5/modeling_t5.py#L248

    assert A.dim() == 3 and B.dim() == 3, (
        "Internal error: T5LayerNorm receiving 2-D tensors, but expected 3-D tensors (sequential inputs)."
    )

    grad_sample = (A * torch.rsqrt(A.pow(2).mean(-1, keepdim=True) + layer.variance_epsilon) * B).sum(dim=1)
    layer.weight.norm_sample = grad_sample.norm(2, dim=1)

#% compute clipped weight gradient    
def _clip_linear_grad(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, C) -> None:
    try:
        grad_weight = torch.einsum('b...,b->...',layer.weight.grad_sample,C)
        del layer.weight.grad_sample
    except:
        grad_weight = torch.einsum('b...d,b...p->pd',A,B)
    return grad_weight

def _clip_normalization_grad(layer, A: torch.Tensor, B: torch.Tensor, C) -> None:
    grad_weight = torch.einsum('b...,b->...',layer.weight.grad_sample,C)
    del layer.weight.grad_sample
    return grad_weight
        
def _clip_embedding_grad(layer: nn.Embedding, A: torch.Tensor, B: torch.Tensor, C) -> None:
    A = F.one_hot(A, num_classes=layer.weight.shape[0]).to(B)  # (batch_size, seq_len, vocab_dim,)
    grad_weight = torch.einsum('b...d,b...p->dp',A,B)
    ## `torch.nn.Embedding` layers don't accumulate gradient on the padding_idx position.
    ##   We do the same for `grad_sample`.
    if layer.padding_idx is not None:
        # `grad_sample` has size (batch_size, num_vocab, embedding_dim).
        grad_weight[layer.padding_idx, :] = 0.
    return grad_weight
                  
def _clip_Conv1D_grad(layer: transformers.pytorch_utils.Conv1D, A: torch.Tensor, B: torch.Tensor, C) -> None:
    try:
        grad_weight = torch.einsum('b...,b->...',layer.weight.grad_sample,C)
        del layer.weight.grad_sample
    except:
        grad_weight = torch.einsum('b...d,b...p->dp',A,B)
    return grad_weight

def _clip_conv_grad(layer, A: torch.Tensor, B: torch.Tensor, C):
    B = B.flatten(2)                                  # F^{-1}(dL/ds)
    try:
        grad_weight = torch.einsum('b...,b->...',layer.weight.grad_sample,C)
        del layer.weight.grad_sample
    except:
        if type(layer)==nn.Conv1d:
            padding = layer.padding if isinstance(
                    layer.padding, tuple) else (*layer.padding, *layer.padding)
            # padded_A = F.pad(A, padding)
            A = F.unfold(A.unsqueeze(-2), kernel_size=(1, *layer.kernel_size),
                                padding=(0, *padding),
                                dilation=(1, *layer.dilation),
                                stride=(1, *layer.stride))
        elif type(layer)==nn.Conv2d:
            A = F.unfold(A, kernel_size=layer.kernel_size,
                                    dilation=layer.dilation, padding=layer.padding,
                                    stride=layer.stride) # U(a)
        elif type(layer)==nn.Conv3d:
            from opacus.utils import tensor_utils
            A = tensor_utils.unfold3d(A, kernel_size=layer.kernel_size,
                                             dilation=layer.dilation, padding=layer.padding,
                                             stride=layer.stride)
        
        grad_weight = torch.einsum('bDT,bpT->pD',A,B)
        #grad_weight = torch.bmm(B, A.permute(0, 2, 1)).sum(dim=0)      

    grad_weight=grad_weight.view(-1, *layer.weight.shape)[0]
    return grad_weight

def _clip_t5_layer_norm_grad(layer: T5LayerNorm, A: torch.Tensor, B: torch.Tensor, clipping_mode: str):
    grad_weight = (A * torch.rsqrt(A.pow(2).mean(-1, keepdim=True) + layer.variance_epsilon) * B).sum(dim=1)
    return grad_weight


_supported_layers_norm_sample_AND_clipping = {
    nn.Embedding: (_compute_embedding_grad_sample, _clip_embedding_grad),
    nn.Linear: (_compute_linear_grad_sample, _clip_linear_grad),
    nn.Conv1d: (_compute_conv_grad_sample, _clip_conv_grad),
    nn.Conv2d: (_compute_conv_grad_sample, _clip_conv_grad),
    nn.LayerNorm: (_compute_layer_norm_grad_sample, _clip_normalization_grad),
    nn.GroupNorm: (_compute_group_norm_grad_sample, _clip_normalization_grad),
    nn.InstanceNorm1d: (_compute_instance_norm_grad_sample, _clip_normalization_grad),
    nn.InstanceNorm2d: (_compute_instance_norm_grad_sample, _clip_normalization_grad),
    nn.InstanceNorm3d: (_compute_instance_norm_grad_sample, _clip_normalization_grad),
    transformers.pytorch_utils.Conv1D: (_compute_Conv1D_grad_sample, _clip_Conv1D_grad),# Conv1D's weight is transposed to nn.Linear's, but this does not matter for the norm
    transformers.models.t5.modeling_t5.T5LayerNorm: (_compute_t5_layer_norm_grad_sample, _clip_t5_layer_norm_grad),
}

#%  we need a new attribute param.private_grad to avoid contamination from non-private .grad
#  we use param.private_grad stores either noise+first micro-batch summed_clipped_grad or only summed_clipped_grad
# note DeepSpeed will not accumulate attribute of param, so param.private_grad does not +=
def _create_or_extend_private_grad(param: torch.Tensor, summed_clipped_grad: torch.Tensor, accumulate_private_grad = True) -> None:
    """Adds summed clipped gradient (not per-sample) to param.private_grad or accumulate the existing tensor."""
    from decimal import Decimal
    config.count_gaussian += 1
    assert summed_clipped_grad.shape == param.shape, f"summed clipped grad.size()={summed_clipped_grad.size()}, param.size()={param.size()}"
    if hasattr(param, "private_grad"):
      if accumulate_private_grad == True:
        param.private_grad += summed_clipped_grad.detach()
      else:
        param.private_grad = summed_clipped_grad.detach()
    else:
        shape_tuple = tuple(int(dim) for dim in param.size())
        start = Decimal(time.time())
        noise = sample_discrete_gaussian(scale=1, shape=shape_tuple, dtype=tf.int32)
        np_array = noise.numpy()
        # Convert NumPy array to a PyTorch tensor
        # to the same device.
        param.private_grad = summed_clipped_grad.detach()+ torch.tensor(np_array).to(summed_clipped_grad.device)  
        end = Decimal(time.time())
        config.global_gaussian = config.global_gaussian + end - start
        config.count_total_gaussian = np.int64(tf.size(noise).numpy()) + np.int64(config.count_total_gaussian)
        
    