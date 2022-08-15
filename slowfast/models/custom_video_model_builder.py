#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""
import torch
import torch.nn as nn

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper, conv_helper
from .build import MODEL_REGISTRY

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}

@MODEL_REGISTRY.register()
class ResNetEnvJoint(nn.Module):
    """
    Joint Modeling, No Sampling
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetEnvJoint, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self.temperature = 2.0
        self.eps=1e-6
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32+width_per_group * 16],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.conv3d1 = conv_helper.Unit3Dpy(
        25, 512,  kernel_size=(3, 3, 3),stride=(2, 1, 1))
        self.bn1 = nn.BatchNorm3d(512)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        self.conv3d2 = conv_helper.Unit3Dpy(
        512, 1024, kernel_size=(1, 3, 3))
        self.bn2 = nn.BatchNorm3d(1024)

        self.conv3d3 = conv_helper.Unit3Dpy(
        1024, 1024, kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(1024)

        self.atten_conv1 = conv_helper.Unit3Dpy(
        1024, 512,  kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.atten_conv2 = conv_helper.Unit3Dpy(
        512, 1,  kernel_size=(1, 3, 3))

    def atten_softmax_3d(self, logits):
        batch_size, T = logits.shape[0], logits.shape[2]
        H, W = logits.shape[3], logits.shape[4]
        # reshape -> softmax (dim=-1) -> reshape back
        logits = logits.view(batch_size, -1, T, H*W)
        atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
        atten_map = atten_map.view(batch_size, -1, T, H, W)
        return atten_map

    def gumbel_softmax_3d(self, logits, x,training):
        batch_size, T = logits.shape[0], logits.shape[2]
        H, W = logits.shape[3], logits.shape[4]
        # gumbel softmax sampling (different mask for each feature channel)
        if training:
            U = torch.zeros_like(x).uniform_()
            gumbel_noise = torch.log(-torch.log(U + self.eps) + self.eps)
            logits = logits - gumbel_noise.detach_()
        # reshape -> softmax (dim=-1) -> reshape back
        logits = logits.view(batch_size, -1, T, H*W)
        atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
        atten_map = atten_map.view(batch_size, -1, T, H, W)
        return atten_map

    def forward(self, inputs_list, bboxes=None):

        x = inputs_list[0]
        print(x[0].size())
        #8x64x8x56x56
        x = self.s1(x)
        print(x[0].size())
        #8x256x8x56x56
        x = self.s2(x)
        print(x[0].size())
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        #8x512x4x28x28
        x = self.s3(x)
        print(x[0].size())
        atten_in = x[0]
        #8x1024x4x14x14
        x = self.s4(x)
        print(x[0].size())
        #8x2048x4x7x7
        x = self.s5(x)
        print(x[0].size())
        #8x64x8x28x28
        prior=inputs_list[1]
        #8x512x4x28x28
        prior = self.conv3d1(prior)
        prior = self.bn1(prior)

        atten_in = torch.cat([atten_in,prior],1)

        atten_in = self.atten_conv1(atten_in)
        atten_logits = self.atten_conv2(atten_in)
        #8x512x4x14x14
        prior = self.maxpool1(prior)
        #8x1024x4x14x14
        prior = self.conv3d2(prior)
        prior = self.bn2(prior)
        # atten_map_prob = self.gumbel_softmax_3d(atten_logits,prior,training)
        atten_map = self.atten_softmax_3d(atten_logits)
        output_map = atten_map
        atten_map = atten_map.repeat(1,prior.size()[1],1,1,1)
        # print(atten_map.size())
        prior=prior*atten_map
        #8x1024x4x7x7
        prior = self.conv3d3(prior)
        prior = self.bn3(prior)
        x[0] = torch.cat([x[0],prior],1)

        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x,output_map

@MODEL_REGISTRY.register()
class ResNetEnvProb(nn.Module):
    """
    Joint Modeling with Sampling
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetEnvProb, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self.temperature = 1.0
        #vis use 1.0, train use 2.0
        self.eps=1e-6
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        if cfg.DATA.EnvType == 'Affordance':
            num_channels = cfg.MODEL.NUM_CLASSES
        elif cfg.DATA.EnvType == 'Semantic':
            num_channels = cfg.DATA.Resolution*cfg.DATA.Resolution*cfg.DATA.Resolution

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32+width_per_group * 16],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.conv3d0 = conv_helper.Unit3Dpy(
        num_channels,256,  kernel_size=(3, 3, 3),stride=(2, 1, 1))
        self.bn0 = nn.BatchNorm3d(256)
        self.conv3d1 = conv_helper.Unit3Dpy(
        256, 512,  kernel_size=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(512)

        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        self.conv3d2 = conv_helper.Unit3Dpy(
        512, 1024, kernel_size=(1, 3, 3))
        self.bn2 = nn.BatchNorm3d(1024)

        self.conv3d3 = conv_helper.Unit3Dpy(
        1024, 1024, kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(1024)

        self.atten_conv1 = conv_helper.Unit3Dpy(
        1024, 512,  kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.atten_bn1 = nn.BatchNorm3d(512)
        self.atten_conv2 = conv_helper.Unit3Dpy(
        512, 1,  kernel_size=(1, 3, 3))
        self.atten_bn2 = nn.BatchNorm3d(1)

    def atten_softmax_3d(self, logits):
        batch_size, T = logits.shape[0], logits.shape[2]
        H, W = logits.shape[3], logits.shape[4]
        # reshape -> softmax (dim=-1) -> reshape back
        logits = logits.view(batch_size, -1, T, H*W)
        atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
        atten_map = atten_map.view(batch_size, -1, T, H, W)
        return atten_map

    def gumbel_softmax_3d(self, logits, x,training):
        batch_size, T = logits.shape[0], logits.shape[2]
        H, W = logits.shape[3], logits.shape[4]
        # gumbel softmax sampling (different mask for each feature channel)
        if training:
            U = torch.zeros_like(x).uniform_()
            gumbel_noise = torch.log(-torch.log(U + self.eps) + self.eps)
            logits = logits - gumbel_noise.detach_()
        # reshape -> softmax (dim=-1) -> reshape back

        logits = logits.view(batch_size, -1, T, H*W)
        atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
        atten_map = atten_map.view(batch_size, -1, T, H, W)
        return atten_map

    def forward(self, inputs_list, bboxes=None, training=True):

        x = inputs_list[0]

        #8x64x8x56x56
        x = self.s1(x)

        #8x256x8x56x56
        x = self.s2(x)

        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        #8x512x4x28x28
        x = self.s3(x)

        atten_in = x[0]
        #8x1024x4x14x14
        x = self.s4(x)

        #8x2048x4x7x7
        x = self.s5(x)

        #8x64x8x28x28
        prior=inputs_list[1]
        #8x512x4x28x28
        prior = self.conv3d0(prior)
        prior = self.bn0(prior)
        prior = self.conv3d1(prior)
        prior = self.bn1(prior)

        atten_in = torch.cat([atten_in,prior],1)
        print(atten_in.size())
        atten_in = self.atten_conv1(atten_in)
        atten_in = self.atten_bn1(atten_in)
        print(atten_in.size())
        atten_logits = self.atten_conv2(atten_in)
        atten_logits = self.atten_bn2(atten_logits)
        #8x512x4x14x14
        prior = self.maxpool1(prior)
        #8x1024x4x14x14
        prior = self.conv3d2(prior)
        prior = self.bn2(prior)
        atten_map_prob = self.gumbel_softmax_3d(atten_logits,prior,training)
        atten_map = self.atten_softmax_3d(atten_logits)

        prior=prior*atten_map_prob*7.0*7.0
        #8x1024x4x7x7
        prior = self.conv3d3(prior)
        prior = self.bn3(prior)
        x[0] = torch.cat([x[0],prior],1)

        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x,atten_map

@MODEL_REGISTRY.register()
class ResNetEnvSimple(nn.Module):
    """
    Joint modeling, location prior as additional inputs, action region not learnable
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetEnvSimple, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        if cfg.DATA.EnvType == 'Affordance':
            num_channels = cfg.MODEL.NUM_CLASSES
        elif cfg.DATA.EnvType == 'SemanticVoxel':
            num_channels = cfg.DATA.Resolution*cfg.DATA.Resolution*cfg.DATA.Resolution
        else:
            num_channels = cfg.DATA.Resolution*cfg.DATA.Resolution*cfg.DATA.Resolution + cfg.MODEL.NUM_CLASSES

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32+width_per_group * 16],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.conv3d1 = conv_helper.Unit3Dpy(
        num_channels, 512,  kernel_size=(3, 3, 3),stride=(2, 1, 1))
        self.bn1 = nn.BatchNorm3d(512)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        self.conv3d2 = conv_helper.Unit3Dpy(
        512, 1024, kernel_size=(1, 3, 3))
        self.bn2 = nn.BatchNorm3d(1024)

        self.conv3d3 = conv_helper.Unit3Dpy(
        1024, 1024, kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(1024)



    def forward(self, inputs_list, bboxes=None):
        x = inputs_list[0]
        x = self.s1(x)

        location_prior = inputs_list[2]

        x = self.s2(x)
        prior=inputs_list[1]

        prior = self.conv3d1(prior)
        prior = self.bn1(prior)
        prior = self.maxpool1(prior)

        location_prior = location_prior.repeat(1,prior.size()[1],1,1,1)
        prior = prior*location_prior

        prior = self.conv3d2(prior)
        prior = self.bn2(prior)

        prior = self.conv3d3(prior)
        prior = self.bn3(prior)

        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)

        x = self.s4(x)

        x = self.s5(x)

        x[0] = torch.cat([x[0],prior],1)

        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x

@MODEL_REGISTRY.register()
class ResNetEnvVanilla(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetEnvVanilla, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        if cfg.DATA.EnvType == 'Affordance':
            num_channels = cfg.MODEL.NUM_CLASSES
        elif cfg.DATA.EnvType == 'SemanticVoxel':
            num_channels = cfg.DATA.Resolution*cfg.DATA.Resolution*cfg.DATA.Resolution
        else:
            num_channels = cfg.DATA.Resolution*cfg.DATA.Resolution*cfg.DATA.Resolution + cfg.MODEL.NUM_CLASSES

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 16],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.conv3d1 = conv_helper.Unit3Dpy(
        num_channels, 512,  kernel_size=(3, 3, 3),stride=(2, 1, 1))
        self.bn1 = nn.BatchNorm3d(512)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        self.conv3d2 = conv_helper.Unit3Dpy(
        512, 1024, kernel_size=(1, 3, 3))
        self.bn2 = nn.BatchNorm3d(1024)

        self.conv3d3 = conv_helper.Unit3Dpy(
        1024, 1024, kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(1024)



    def forward(self, inputs_list, bboxes=None):
        location_prior = inputs_list[2]
        prior=inputs_list[1]
        prior = self.conv3d1(prior)
        prior = self.bn1(prior)
        prior = self.maxpool1(prior)

        location_prior = location_prior.repeat(1,prior.size()[1],1,1,1)
        prior = prior*location_prior

        prior = self.conv3d2(prior)
        prior = self.bn2(prior)

        prior = self.conv3d3(prior)
        prior = self.bn3(prior)

        x = prior

        if self.enable_detection:
            x = self.head([x], bboxes)
        else:
            x = self.head([x])
        return x

@MODEL_REGISTRY.register()
class ResNetJointProb(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetJointProb, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self.temperature = 2.0
        self.eps=1e-6
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()


        affordance_channels = cfg.MODEL.NUM_CLASSES

        semantic_channels = cfg.DATA.Resolution*cfg.DATA.Resolution*cfg.DATA.Resolution

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32+width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.affordance_conv3d1 = conv_helper.Unit3Dpy(
        affordance_channels, 512,  kernel_size=(3, 3, 3),stride=(2, 1, 1))
        self.affordance_bn1 = nn.BatchNorm3d(512)
        self.affordance_maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        self.affordance_conv3d2 = conv_helper.Unit3Dpy(
        512, 1024, kernel_size=(1, 3, 3))
        self.affordance_bn2 = nn.BatchNorm3d(1024)

        self.affordance_conv3d3 = conv_helper.Unit3Dpy(
        1024, 1024, kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.affordance_bn3 = nn.BatchNorm3d(1024)

        self.semantic_conv3d1 = conv_helper.Unit3Dpy(
        semantic_channels, 512,  kernel_size=(3, 3, 3),stride=(2, 1, 1))
        self.semantic_bn1 = nn.BatchNorm3d(512)
        self.semantic_maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        self.semantic_conv3d2 = conv_helper.Unit3Dpy(
        512, 1024, kernel_size=(1, 3, 3))
        self.semantic_bn2 = nn.BatchNorm3d(1024)

        self.semantic_conv3d3 = conv_helper.Unit3Dpy(
        1024, 1024, kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.semantic_bn3 = nn.BatchNorm3d(1024)

        self.atten_conv1 = conv_helper.Unit3Dpy(
        1536, 512,  kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.atten_conv2 = conv_helper.Unit3Dpy(
        512, 1,  kernel_size=(1, 3, 3))

    def atten_softmax_3d(self, logits):
        batch_size, T = logits.shape[0], logits.shape[2]
        H, W = logits.shape[3], logits.shape[4]
        # reshape -> softmax (dim=-1) -> reshape back
        logits = logits.view(batch_size, -1, T, H*W)
        atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
        atten_map = atten_map.view(batch_size, -1, T, H, W)
        return atten_map

    def gumbel_softmax_3d(self, logits, x,training):
        batch_size, T = logits.shape[0], logits.shape[2]
        H, W = logits.shape[3], logits.shape[4]
        # gumbel softmax sampling (different mask for each feature channel)
        if training:
            U = torch.zeros_like(x).uniform_()
            gumbel_noise = torch.log(-torch.log(U + self.eps) + self.eps)
            logits = logits - gumbel_noise.detach_()
        # reshape -> softmax (dim=-1) -> reshape back

        logits = logits.view(batch_size, -1, T, H*W)
        atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
        atten_map = atten_map.view(batch_size, -1, T, H, W)
        return atten_map

    def forward(self, inputs_list, bboxes=None,training=True):

        x = inputs_list[0]
        #8x64x8x56x56
        x = self.s1(x)
        #8x256x8x56x56
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        #8x512x4x28x28
        x = self.s3(x)
        atten_in = x[0]
        #8x1024x4x14x14
        x = self.s4(x)
        #8x2048x4x7x7
        x = self.s5(x)

        #8x64x8x28x28
        affordance_prior=inputs_list[1]
        #8x512x4x28x28
        affordance_prior = self.affordance_conv3d1(affordance_prior)
        affordance_prior = self.affordance_bn1(affordance_prior)

        semantic_prior=inputs_list[2]
        #8x512x4x28x28
        semantic_prior = self.semantic_conv3d1(semantic_prior)
        semantic_prior = self.semantic_bn1(semantic_prior)

        atten_in = torch.cat([atten_in,affordance_prior,semantic_prior],1)
        atten_in = self.atten_conv1(atten_in)
        atten_logits = self.atten_conv2(atten_in)

        #8x512x4x14x14
        affordance_prior = self.affordance_maxpool1(affordance_prior)
        #8x1024x4x14x14
        affordance_prior = self.affordance_conv3d2(affordance_prior)
        affordance_prior = self.affordance_bn2(affordance_prior)

        #8x512x4x14x14
        semantic_prior = self.semantic_maxpool1(semantic_prior)
        #8x1024x4x14x14
        semantic_prior = self.semantic_conv3d2(semantic_prior)
        semantic_prior = self.semantic_bn2(semantic_prior)

        atten_map_prob_affordance = self.gumbel_softmax_3d(atten_logits,affordance_prior,training)
        atten_map_prob_semantic = self.gumbel_softmax_3d(atten_logits,semantic_prior,training)

        atten_map = self.atten_softmax_3d(atten_logits)

        affordance_prior=affordance_prior*atten_map_prob_affordance
        #8x1024x4x7x7
        affordance_prior = self.affordance_conv3d3(affordance_prior)
        affordance_prior = self.affordance_bn3(affordance_prior)

        semantic_prior=semantic_prior*atten_map_prob_semantic
        #8x1024x4x7x7
        semantic_prior = self.semantic_conv3d3(semantic_prior)
        semantic_prior = self.semantic_bn3(semantic_prior)

        x[0] = torch.cat([x[0],affordance_prior,semantic_prior],1)

        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x,atten_map

@MODEL_REGISTRY.register()
class ResNetEnvBase(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetEnvBase, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self.temperature = 2.0
        self.eps=1e-6
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        if cfg.DATA.EnvType == 'Affordance':
            num_channels = cfg.MODEL.NUM_CLASSES
        elif cfg.DATA.EnvType == 'SemanticVoxel':
            num_channels = cfg.DATA.Resolution*cfg.DATA.Resolution*cfg.DATA.Resolution
        else:
            num_channels = cfg.DATA.Resolution*cfg.DATA.Resolution*cfg.DATA.Resolution + cfg.MODEL.NUM_CLASSES

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.atten_conv1 = conv_helper.Unit3Dpy(
        512, 256,  kernel_size=(1, 3, 3),stride=(1, 2, 2))
        self.atten_conv2 = conv_helper.Unit3Dpy(
        256, 1,  kernel_size=(1, 3, 3))


    def atten_softmax_3d(self, logits):
        batch_size, T = logits.shape[0], logits.shape[2]
        H, W = logits.shape[3], logits.shape[4]
        # reshape -> softmax (dim=-1) -> reshape back
        logits = logits.view(batch_size, -1, T, H*W)
        atten_map = nn.functional.softmax(logits / self.temperature, dim=-1)
        atten_map = atten_map.view(batch_size, -1, T, H, W)
        return atten_map

    def forward(self, inputs_list, bboxes=None):
        x = inputs_list[0]
        # print(x[0].size())
        #8x64x8x56x56
        x = self.s1(x)
        #8x256x8x56x56
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        #8x512x4x28x28
        x = self.s3(x)
        atten_in = x[0]
        #8x1024x4x14x14
        x = self.s4(x)
        #8x2048x4x7x7
        x = self.s5(x)

        atten_in = self.atten_conv1(atten_in)
        atten_logits = self.atten_conv2(atten_in)
        atten_map = self.atten_softmax_3d(atten_logits)

        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x,atten_map
