import torch
import argparse
import numpy as np
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

# custom modules
from models import Neuraldecipher
from utils import EarlyStopping, create_train_and_test_set, create_data_loaders, str_to_bool
from trainer import Trainer
from losses import LogCoshLoss, XTanhLoss, XSigmoidLoss, CosineSimLoss


def get_args():
    """
    Argparse helper function
    :return:
    """
    parser = argparse.ArgumentParser(description='Main script to train the Neuraldecipher Model')

    parser.add_argument('--config', action='store', dest='config',
                        default='../params/1024_config_bit_gpu.json',
                        help='Configuration for training. \
                              Default: params/1024_config_bit_gpu.json', type=str)

    parser.add_argument('--split', action='store', dest='split', default='cluster',
                        help='Insert "cluster" or "random" for train and validation set creation. \
                             Default: "cluster".', type=str)

    parser.add_argument('--workers', action='store', dest='num_workers', default=5,
                        help='Number of workers for dataloader. \
                                Default: 5.', type=int)

    parser.add_argument('--cosineloss', action='store', dest='cosineloss', default='False',
                        help='Whether or not to use cosine similarity loss in training. Defaults to False')


    args = parser.parse_args()

    return args


def main():
    """
    Main call function for the script
    :return:
    """

    import os
    source_path = os.path.abspath(__file__)
    base_path = os.path.dirname(os.path.dirname(source_path))

    args = get_args()
    if "params/" not in args.config:
        args.config = "../params/" + args.config

    with open(os.path.join(base_path,args.config), 'r', encoding='utf-8') as config_file:
        json_string = config_file.read()

    if args.split.lower() == 'cluster':
        random_split = False
        print("Using cluster split for train and validation")
    else:
        random_split = True
        print("Using random split for train and validation")

    import os
    source_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(source_path))

    params = json.loads(json_string)
    print('Neuraldecipher training with param settings:')
    print(params)

    if params['neuraldecipher'].get('norm_before') is None:
        params['neuraldecipher']['norm_before'] = True

    # instantiate neuraldecipher model
    neuraldecipher = Neuraldecipher(**params['neuraldecipher'])
    print("Neuraldecipher model:")
    print(neuraldecipher)

    # instantiate trainer object
    trainer = Trainer(model=neuraldecipher, trainparams=params['training'])
    earlystopping = EarlyStopping(mode='min', patience=params['training']['patience'])
    optimizer = torch.optim.Adam(params=neuraldecipher.parameters(),
                                 betas=(params['training']['b1'], params['training']['b2']),
                                 lr=params['training']['lr'],
                                 weight_decay=params['training']['weight_decay'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.70, patience=10, verbose=True)

    if params['training']['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif params['training']['loss'] == 'x-sigmoid':
        criterion = XSigmoidLoss()
    elif params['training']['loss'] == 'x-tanh':
        criterion = XTanhLoss()
    elif params['training']['loss'] == 'log-cosh':
        criterion = LogCoshLoss()
    else:
        criterion = torch.nn.MSELoss()

    if str_to_bool(args.cosineloss):
        criterion_2 = CosineSimLoss()
        criteria = [criterion, criterion_2]
        print("Using {} and cosine difference loss.".format(params['training']['loss']))
    else:
        criteria = [criterion]
        print("Using {} loss.".format(params['training']['loss']))

    seed = params['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if 'cuda' in params['training']['device']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # obtain datasets for training and validation
    train_data, test_data = create_train_and_test_set(ecfp_path=params['training']['data_dir'],
                                                      test_group=7,
                                                      random_split=random_split)
    # create dataloaders
    train_loader, test_loader = create_data_loaders(train_data, test_data,
                                                    batch_size=params['training']['batch_size'],
                                                    num_workers=args.num_workers)

    trainer._train(criteria, earlystopping, scheduler, optimizer, train_loader, test_loader, verbose=True)


# main
if __name__ == "__main__":
    main()