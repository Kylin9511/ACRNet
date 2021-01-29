import time
import os
import torch
from collections import namedtuple

from utils import logger
from utils.statics import AverageMeter, evaluator

__all__ = ['Tester']


field = ('nmse', 'rho', 'epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Tester:
    r""" The testing interface for classification
    """

    def __init__(self, model, device, criterion, print_freq=10):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            loss, rho, nmse = self._iteration(test_data)
        if verbose:
            print(f'\n=> Test result: \nloss: {loss:.4e}'
                  f'    rho: {rho:.4f}    NMSE: {nmse:.4f}\n')
        return loss, rho, nmse

    def _iteration(self, data_loader):
        r""" protected function which test the model on given data loader for one epoch.
        """

        iter_rho = AverageMeter('Iter rho')
        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (sparse_gt, raw_gt) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            raw_gt = raw_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)
            rho, nmse = evaluator(sparse_pred, sparse_gt, raw_gt)

            # Log and visdom update
            iter_loss.update(loss)
            iter_rho.update(rho)
            iter_nmse.update(nmse)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.4e} | '
                            f'rho: {iter_rho.avg:.4f} | '
                            f'NMSE: {10 * torch.log10(iter_nmse.avg):.4f} | '
                            f'time: {iter_time.avg:.3f}')

        nmse = 10 * torch.log10(iter_nmse.avg)  # global NMSE in dB
        logger.info(f'=> Test rho:{iter_rho.avg:.4f}  NMSE: {nmse:.4f}\n')

        return iter_loss.avg, iter_rho.avg, nmse
