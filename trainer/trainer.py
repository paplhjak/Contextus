import os
import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from utils import plot_depth_from_x
from utils import cityscapes_labels
from utils import plot_semantic_results
from utils import plot_depth_completion_results
from PIL import Image


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        if not os.path.exists(str(self.checkpoint_dir) + '/epoch_' + str(epoch) + '/'):
            os.makedirs(str(self.checkpoint_dir) + '/epoch_' + str(epoch) + '/')

        self.model.to(self.device)
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, rgb, target) in enumerate(tqdm(self.data_loader)):
            self.optimizer.zero_grad()
            data, rgb, target = data.to(self.device, non_blocking=True), rgb.to(self.device,
                                                                                non_blocking=True), target.to(
                self.device, non_blocking=True)

            """
            get output and loss
            """

            output = self.model(data, rgb)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            """
            create visualization
            """
            if (batch_idx % self.config['save_image_every_n_epochs'] == self.config['save_image_every_n_epochs'] - 1):
                im = plot_depth_completion_results.create_vis(rgb[0].detach().cpu(), target[0].detach().cpu(),
                                                              output[0].detach().cpu(), data[0].detach().cpu())
                im.save(str(self.checkpoint_dir) + '/epoch_' + str(epoch) + '/TRAIN_' + str(batch_idx) + '.png')

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx, 'train')
            self.train_metrics.update('loss', loss.item())
            for met_idx in range(len(self.metric_ftns)):
                met = self.metric_ftns[met_idx]
                metric_result = float(met(output.detach(), target.detach()))
                self.train_metrics.update(met.__name__, metric_result)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
               
               

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, rgb, target) in enumerate(tqdm(self.valid_data_loader)):

                data, rgb, target = data.to(self.device), rgb.to(self.device), target.to(self.device)
                output = self.model(data, rgb)
                loss = self.criterion(output, target)

                if (batch_idx % self.config['save_image_every_n_epochs'] == self.config[
                    'save_image_every_n_epochs'] - 1):
                    im = plot_depth_completion_results.create_vis(rgb[0].detach().cpu(), target[0].detach().cpu(),
                                                                  output[0].detach().cpu(), data[0].detach().cpu())
                    im.save(str(self.checkpoint_dir) + '/epoch_' + str(epoch) + '/VAL_' + str(batch_idx) + '.png')
                    # plot_depth_from_x.plot_depth_from_tensor(output[0], str(self.checkpoint_dir) + '/epoch_' + str(
                    #    epoch) + '/VAL_' + str(batch_idx) + '.png')

                self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met_idx in range(len(self.metric_ftns)):
                    met = self.metric_ftns[met_idx]
                    metric_result = float(met(output.detach(), target.detach()))
                    self.valid_metrics.update(met.__name__, metric_result)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def profile(self):
        import torch.autograd.profiler as profiler
        with profiler.profile(use_cuda=True) as prof:
            self.model.to(self.device)
            self.model.train()
    
            for batch_idx, (data, rgb, target) in enumerate(tqdm(self.data_loader)):
                if batch_idx == 30:
                    break
                    
                self.optimizer.zero_grad()
                data, rgb, target = data.to(self.device, non_blocking=True), rgb.to(self.device,
                                                                                    non_blocking=True), target.to(
                    self.device, non_blocking=True)
        
                output = self.model(data, rgb)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))