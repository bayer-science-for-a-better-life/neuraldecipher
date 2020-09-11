import torch
import os
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# custom modules
from metrics import l2_distance
from models import Neuraldecipher

# for type checking
from utils import EarlyStopping
import typing

class Trainer:
    """
    Trainer object that contains the Neuraldecipher model for training.
    """

    def __init__(self, model: Neuraldecipher, trainparams: dict):
        self.model = model
        self.trainparams = trainparams
        self.device = torch.device(self.trainparams['device'])
        self.model = self.model.to(self.device)

    def _train(self, criterion: typing.Callable,
               earlystopping: EarlyStopping,
               scheduler: torch.optim.lr_scheduler,
               optimizer: torch.optim.Optimizer,
               train_loader: torch.utils.data.DataLoader,
               valid_loader=typing.Union[None, torch.utils.data.DataLoader],
               verbose: bool = True):
        """
        Class function to train the Trainer object. After every training epoch, a validation can be performed.
        :param criterion: (torch.nn) loss class
        :param earlystopping: (EarlyStopping) custom class for doing early stopping
        :param scheduler: (torch.optim.lr_scheduler) for adjusting learning rates during training
        :param optimizer: (torch.optim.Optimizer)
        :param train_loader: (torch.utils.data.DataLoader) object for training
        :param valid_loader: (torch.utils.data.DataLoader) or None in case validation should be performed after every epoch
        :param verbose (bool): whether or not to print out on console. Default: True
        :return: training and validation metrics
        """
        print('Training the Neuraldecipher for {} epochs.'.format(self.trainparams['n_epochs']))
        train_count = 0
        test_count = 0

        # create modelpath savedirs
        logdir_path = os.path.join('../logs', self.trainparams['output_dir'])
        model_outpath = os.path.join('../models', self.trainparams['output_dir'])

        if not os.path.exists(logdir_path):
            os.makedirs(logdir_path)
        if not os.path.exists(model_outpath):
            os.makedirs(model_outpath)

        # create summary writer
        writer = SummaryWriter(log_dir=logdir_path)

        # turn model in train mode
        self.model = self.model.train()

        # saving arrays
        self.train_loss_array = []
        self.test_loss_array = []
        self.train_euclidean_array = []
        self.test_euclidean_array = []

        if len(criterion) == 2:
            # Motivate distance loss and cosine similarity
            a = 10
            weight_func = lambda x: (a ** x - 1) / (a - 1)
            self.cosine_weight_loss = [weight_func(f / self.trainparams['n_epochs']) for f in
                                       range(self.trainparams['n_epochs'])]

        for epoch in range(0, self.trainparams['n_epochs']):

            self.train_loss = 0.0
            self.test_loss = 0.0
            self.train_euclidean = 0.0
            self.test_euclidean = 0.0

            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

                ecfp_in = batch['ecfp'].to(device=self.device, dtype=torch.float32)
                cddd_out = batch['cddd'].to(device=self.device, dtype=torch.float32)

                cddd_predicted = self.model(ecfp_in)

                # hacky solution in case there are more criteria
                if len(criterion) == 1:  # only difference, e.g MSE or logcosh
                    # compute prediction and loss
                    loss = criterion[0](cddd_predicted, cddd_out)
                elif len(criterion) == 2:  # difference AND cosine loss
                    d_loss = criterion[0](cddd_predicted, cddd_out)
                    cosine_loss = 1 - criterion[1](cddd_predicted, cddd_out)
                    loss = d_loss + self.cosine_weight_loss[epoch] * cosine_loss

                batch_train_euclidean = l2_distance(y_pred=cddd_predicted, y_true=cddd_out).data.item()
                self.train_euclidean += batch_train_euclidean

                # compute gradients and update weights
                optimizer.zero_grad()
                loss.backward()
                self.train_loss += loss.data.item()
                optimizer.step()

                writer.add_scalar(tag='Loss/train', scalar_value=loss.data.item(), global_step=train_count)
                writer.add_scalar(tag='Euclidean/train', scalar_value=batch_train_euclidean, global_step=train_count)

                train_count += 1
                if train_count % 500 == 0 and train_count != 0 and verbose:
                    tqdm.write('*' * 100)
                    tqdm.write(
                        'Epoch [%d/%d] Batch [%d/%d] Loss Train: %.4f Mean L2 Distance %.4f' % (
                            epoch, self.trainparams['start_epoch'] + self.trainparams['n_epochs'],
                            step, len(train_loader), loss.data.item(), batch_train_euclidean
                        )
                    )
                    tqdm.write('*' * 100 + '\n')

            # learning rate scheduler at the end of the epoch
            self.train_loss /= len(train_loader)
            self.train_euclidean /= len(train_loader)
            if scheduler:
                scheduler.step(self.train_euclidean)

            # evaluation
            if valid_loader:
                if verbose:
                    tqdm.write('Epoch %d finished. Doing validation:' % (epoch))
                writer, test_count = self._eval(criterion, valid_loader, writer, test_count, epoch)
                self.test_loss /= len(valid_loader)
                self.test_euclidean /= len(valid_loader)

                if verbose:
                    tqdm.write(
                        'Epoch [%d/%d] Loss Train: %.4f Euclidean Train: %.4f Loss Valid: %.4f Euclidean Valid: %.4f' % (
                            epoch, self.trainparams['n_epochs'], self.train_loss, self.train_euclidean,
                            self.test_loss, self.test_euclidean
                        ))

                if earlystopping:
                    earlystopping(metric_val=self.test_euclidean, model=self.model,
                                  modelpath=model_outpath, epoch=epoch)
                    if earlystopping.early_stop:
                        print('Early stopping the training the NeuralDecipher Model on ECFP fingerprints  \
                        with radii {} and {} bit length. \n Results and models are saved at {} and {}.'.format(
                            self.trainparams['radii'], self.model.input_dim, logdir_path, model_outpath)
                        )
                        break

            ## array saving
            self.train_loss_array.append(self.train_loss)
            self.test_loss_array.append(self.test_loss)
            self.train_euclidean_array.append(self.train_euclidean)
            self.test_euclidean_array.append(self.test_euclidean)

        print('Finished training the NeuralDecipher Model on ECFP fingerprints with radii {} and {} bit length. \n \
              Results and models are saved at {} and {}.'.format(
            self.trainparams['radii'], self.model.input_dim, logdir_path, model_outpath)
        )

        ## model saving
        torch.save(self.model.state_dict(),
                   os.path.join(model_outpath, 'final_model_{}.pt'.format(self.test_euclidean)))

        ## array saving
        json_array = {'train_loss': self.train_loss_array,
                      'train_euclidean': self.train_euclidean_array,
                      'test_loss': self.test_loss_array,
                      'test_euclidean': self.test_euclidean_array}


        json_filepath = os.path.join(model_outpath, 'loss_metrics.json')
        with open(json_filepath, 'w') as f:
            json.dump(json_array, f)

    def _eval(self, criterion, valid_loader, writer, test_count, epoch):
        # in case batch-normalization is used, the model should be turned in eval model.
        # therefore we turn on eval model and still use the torch.no_grad() context.
        self.model = self.model.eval()

        cddd_reconstructed = []
        smiles_true = []
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                ecfp_in = batch['ecfp'].to(device=self.device, dtype=torch.float32)
                cddd_out = batch['cddd'].to(device=self.device, dtype=torch.float32)
                smiles_true.append(batch['smiles'])

                cddd_predicted = self.model(ecfp_in)
                cddd_reconstructed.append(cddd_predicted.data.cpu().numpy().tolist())

                # hacky solution in case there are more criteria
                if len(criterion) == 1:  # only difference, e.g MSE or logcosh
                    loss = criterion[0](cddd_predicted, cddd_out)
                elif len(criterion) == 2:  # difference AND cosine loss
                    d_loss = criterion[0](cddd_predicted, cddd_out)
                    cosine_loss = 1 - criterion[1](cddd_predicted, cddd_out)
                    loss = d_loss + self.cosine_weight_loss[epoch] * cosine_loss

                self.test_loss += loss.data.item()
                batch_test_euclidean = l2_distance(y_pred=cddd_predicted, y_true=cddd_out).data.item()
                self.test_euclidean += batch_test_euclidean

                if writer:
                    writer.add_scalar(tag='Loss/test', scalar_value=loss.data.item(), global_step=test_count)
                    writer.add_scalar(tag='Euclidean/test', scalar_value=batch_test_euclidean, global_step=test_count)

                test_count += 1

        # turn model into train mode
        self.model = self.model.train()

        return writer, test_count