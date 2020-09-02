import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.depth_loss as module_depth_loss
import model.semantic_loss as module_semantic_loss
import model.depth_metric as module_depth_metric
import model.semantic_metric as module_semantic_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = config.init_obj('validation_data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    if config['task'] == 'features4depth' or config['task'] == 'rgb4depth':
        criterion = getattr(module_depth_loss, config['loss'])
        metrics = [getattr(module_depth_metric, met) for met in config['metrics']]
    elif config['task'] == 'depth4semantic':
        criterion = getattr(module_semantic_loss, config['loss'])
        metrics = [getattr(module_semantic_metric, met) for met in config['metrics']]
    else:
        raise NotImplementedError("Specified task not implemented.")


    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    
    trainer.train()
    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='DeepLab with Xception backbone')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
