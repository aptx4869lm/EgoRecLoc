#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch.nn as nn

def get_padding_shape(filter_shape, stride, mod=0):
  """Fetch a tuple describing the input padding shape.

  NOTES: To replicate "TF SAME" style padding, the padding shape needs to be
  determined at runtime to handle cases when the input dimension is not divisible
  by the stride.
  See https://stackoverflow.com/a/49842071 for explanation of TF padding logic
  """
  def _pad_top_bottom(filter_dim, stride_val, mod):
    if mod:
      pad_along = max(filter_dim - mod, 0)
    else:
      pad_along = max(filter_dim - stride_val, 0)
    pad_top = pad_along // 2
    pad_bottom = pad_along - pad_top
    return pad_top, pad_bottom

  padding_shape = []
  for idx, (filter_dim, stride_val) in enumerate(zip(filter_shape, stride)):
    depth_mod = (idx == 0) and mod
    pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val, depth_mod)
    padding_shape.append(pad_top)
    padding_shape.append(pad_bottom)

  depth_top = padding_shape.pop(0)
  depth_bottom = padding_shape.pop(0)
  padding_shape.append(depth_top)
  padding_shape.append(depth_bottom)
  return tuple(padding_shape)

def simplify_padding(padding_shapes):
  all_same = True
  padding_init = padding_shapes[0]
  for pad in padding_shapes[1:]:
    if pad != padding_init:
      all_same = False
  return all_same, padding_init

class Unit3Dpy(nn.Module):
  """
  Conv3D + BN3D + Relu
  """
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=(1, 1, 1),
               stride=(1, 1, 1),
               activation='relu',
               padding='SAME',
               use_bias=False,
               use_bn=True):
    super(Unit3Dpy, self).__init__()

    # setup params
    self.padding = padding
    self.activation = activation
    self.use_bn = use_bn
    self.stride = stride

    # follow the padding of tensorflow (somewhat complicated logic here)
    if padding == 'SAME':
      padding_shape = get_padding_shape(kernel_size, stride)
      simplify_pad, pad_size = simplify_padding(padding_shape)
      self.simplify_pad = simplify_pad
      if stride[0] > 1:
        padding_shapes = [get_padding_shape(kernel_size, stride, mod) for
                          mod in range(stride[0])]
      else:
        padding_shapes = [padding_shape]
    elif padding == 'VALID':
      padding_shape = 0
    else:
      raise ValueError(
        'padding should be in [VALID|SAME] but got {}'.format(padding))

    if padding == 'SAME':
      if not simplify_pad:
        # pad - conv
        self.pads = [nn.ConstantPad3d(x, 0) for x in padding_shapes]
        self.conv3d = nn.Conv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                bias=use_bias)
      else:
        self.conv3d = nn.Conv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=pad_size,
                                bias=use_bias)
    elif padding == 'VALID':
      self.conv3d = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              padding=padding_shape,
                              stride=stride,
                              bias=use_bias)
    if self.use_bn:
      tf_style_eps = 1E-3
      self.batch3d = nn.BatchNorm3d(out_channels, eps=tf_style_eps)

    if activation == 'relu':
      self.activation = nn.ReLU(inplace=True)

  def forward(self, inp):
    # pad -> conv3d -> bn -> relu
    if self.padding == 'SAME' and self.simplify_pad is False:
      # Determine the padding to be applied by examining the input shape
      pad_idx = inp.shape[2] % self.stride[0]
      pad_op = self.pads[pad_idx]
      inp = pad_op(inp)
    out = self.conv3d(inp)
    if self.use_bn:
      out = self.batch3d(out)
    if self.activation is not None:
      out = self.activation(out)
    return out
