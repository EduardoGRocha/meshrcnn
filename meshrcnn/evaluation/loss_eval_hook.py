# Adapted from https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
import logging

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(1, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        cls_losses = []
        box_reg_losses = []
        occnet_losses = []
        rpn_cls_losses = []
        rpn_loc_losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            cls_losses.append(loss_batch[0])
            box_reg_losses.append(loss_batch[1])
            occnet_losses.append(loss_batch[2])
            rpn_cls_losses.append(loss_batch[3])
            rpn_loc_losses.append(loss_batch[4])

        mean_loss_cls = np.mean(cls_losses)
        mean_loss_box_reg = np.mean(box_reg_losses)
        mean_loss_occnet = np.mean(occnet_losses)
        mean_loss_rpn_cls = np.mean(rpn_cls_losses)
        mean_loss_rpn_loc = np.mean(rpn_loc_losses)

        self.trainer.storage.put_scalar('val_loss_cls', mean_loss_cls)
        self.trainer.storage.put_scalar('val_loss_box_reg', mean_loss_box_reg)
        self.trainer.storage.put_scalar('val_loss_occnet', mean_loss_occnet)
        self.trainer.storage.put_scalar('val_loss_rpn_cls', mean_loss_rpn_cls)
        self.trainer.storage.put_scalar('val_loss_rpn_loc', mean_loss_rpn_loc)

        # self.trainer.storage.put_scalar('val_loss_cls', 0)
        # self.trainer.storage.put_scalar('val_loss_box_reg', 0)
        # self.trainer.storage.put_scalar('val_loss_occnet', 0)
        # self.trainer.storage.put_scalar('val_loss_rpn_cls', 0)
        # self.trainer.storage.put_scalar('val_loss_rpn_loc', 0)
        comm.synchronize()
        # return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return metrics_dict["loss_cls"], metrics_dict["loss_box_reg"], \
               metrics_dict["loss_occnet"], metrics_dict["loss_rpn_cls"], \
               metrics_dict["loss_rpn_loc"]

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)