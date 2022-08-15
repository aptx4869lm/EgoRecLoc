#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from iopath.common.file_io import g_pathmgr

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)

def to_onehot(indices, num_classes):
    """Convert a tensor of indices of any shape `(N, ...)` to a
    tensor of one-hot indicators of shape `(N, num_classes, ...)`.
    """
    onehot = torch.zeros(indices.shape[0],
                        num_classes,
                        *indices.shape[1:],
                        device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)

def mean_class_accuracy(cm):
    """Compute mean class accuracy based on the input confusion matrix"""
    # Increase floating point precision
    cm = cm.type(torch.float64)
    cls_cnt = cm.sum(dim=1) + 1e-15
    cls_hit = cm.diag()
    cls_acc = (cls_hit/cls_cnt).mean().item()
    return cls_acc

def confusion_matrix(pred, target):
    num_classes = pred.shape[1]
    assert pred.shape[0] == target.shape[0]
    with torch.no_grad():
        target_ohe = to_onehot(target, num_classes)
        target_ohe_t = target_ohe.transpose(0, 1).float()

        pred_idx = torch.argmax(pred, dim=1)
        pred_ohe = to_onehot(pred_idx.reshape(-1), num_classes)
        pred_ohe = pred_ohe.float()

        confusion_matrix = torch.matmul(target_ohe_t, pred_ohe)
    return confusion_matrix

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    attengt_list = []
    attenpred_list = []
    labels_list=[]
    preds_list=[]
    videoid_list=[]
    meta_list=[]
    for cur_iter, (inputs, labels, video_idx, semantic_prior, affordance_prior, location_prior, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            # for key, val in meta.items():
            #     if isinstance(val, (list,)):
            #         for i in range(len(val)):
            #             val[i] = val[i].cuda(non_blocking=True)
            #     else:
            #         meta[key] = val.cuda(non_blocking=True)
            inputs_list=[]
            location_prior = location_prior.cuda()
            semantic_prior = semantic_prior.cuda()
            affordance_prior = affordance_prior.cuda()

        test_meter.data_toc()

        if cfg.DATA.EnvType == 'Affordance':
            inputs_list.append(inputs)
            inputs_list.append(affordance_prior)
            inputs_list.append(location_prior)

            # Perform the forward pass.
        elif cfg.DATA.EnvType == 'Semantic':
            inputs_list.append(inputs)
            inputs_list.append(semantic_prior)
            inputs_list.append(location_prior)
        else:
            inputs_list.append(inputs)

        # Explicitly declare reduction to mean.

        if 'Prob' in cfg.MODEL.MODEL_NAME:
            preds,atten_map = model(inputs_list, training = False)

        elif  'Soft' in cfg.MODEL.MODEL_NAME or 'Base' in cfg.MODEL.MODEL_NAME:
            preds,atten_map = model(inputs_list)

        else:
            preds = model(inputs)

        # Perform the forward pass.
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            # print(atten_map.size())
            preds, labels, video_idx, atten_pred, atten_gt = du.all_gather(
               [preds, labels, video_idx, atten_map, location_prior]
            )
            # preds, labels, video_idx= du.all_gather(
            #     [preds, labels, video_idx])
            attengt_list.append(atten_gt.cpu())
            attenpred_list.append(atten_pred.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds.cpu())
            videoid_list.append(video_idx.cpu())
            # print(atten_gt.size())
        if cfg.NUM_GPUS:
            atten_gt = location_prior.cpu()
            atten_pred = atten_map.cpu()
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()


            attengt_list.append(atten_gt.cpu())
            attenpred_list.append(atten_pred.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds.cpu())
            videoid_list.append(video_idx.cpu())
            meta_list.append(meta)

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()
        # if cur_iter>20:
        #     break

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)
            
            if du.is_root_proc():
                print(save_path)
                with g_pathmgr.open(save_path, "wb") as f:
                    # pickle.dump([preds_list,labels_list, videoid_list], f)
                    print(len(attengt_list))
                    print(len(preds_list))
                    print(len(meta_list))
                    pickle.dump([attengt_list, attenpred_list, preds_list,labels_list, meta_list], f)

                logger.info(
                    "Successfully saved prediction results to {}".format(save_path)
                )
        print(all_labels.size())
        print(all_preds.size())
        cm =confusion_matrix(all_preds,all_labels)
        cls_acc = mean_class_accuracy(cm)
        print('***Mean Cls Acc {cls_acc:.3f}'.format(cls_acc=100*cls_acc))
    test_meter.finalize_metrics()
    test_meter.reset()


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        print(len(test_loader.dataset))
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
