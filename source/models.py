import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from collections import OrderedDict


class Neuraldecipher(nn.Module):
    def __init__(self, input_dim: int = 1024, layers: list = [1024, 512],
                 normalization='batch', dropout: float = 0.0, use_tanh: bool = True,
                 activation: str = 'relu', output_dim: int = 512, norm_before: bool = True):
        super(Neuraldecipher, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = [self.input_dim] + layers + [self.output_dim]
        self.normalization = normalization
        self.dropout = dropout
        self.use_tanh = use_tanh
        self.activation = activation.lower().strip()
        self.norm_before = norm_before
        assert self.activation in ['', 'relu', 'softplus', 'leaky_relu', 'elu', 'selu']
        assert self.normalization in ['', 'batch', 'layer', 'weight']

        self.model = self._create_model()

    # Define torch building blocks
    def _block(self, in_feat: int, out_feat: int):
        '''
        Defines a torch operations on a linear layer
        :param in_feat: Number of input features for the layer
        :param out_feat: Number of output features for the layer
        :return: list of torch operations to apply at one layer
        '''

        # Affine linear transformation
        if self.normalization == 'weight':
            layer = {
                'weight_norm_linear': weight_norm(module=nn.Linear(in_features=in_feat, out_features=out_feat))}
        else:
            layer = {'linear': nn.Linear(in_features=in_feat, out_features=out_feat)}

        if self.dropout > 0:
            layer.update({'dropout': nn.Dropout(p=self.dropout)})

        # This code should be improved...
        if self.norm_before:  # Normalization before the Non-Linearity
            if self.normalization == 'batch':
                layer.update({'batch_normalization': nn.BatchNorm1d(num_features=out_feat, eps=1e-05, momentum=0.1,
                                                                    affine=True, track_running_stats=True)})
            elif self.normalization == 'layer':
                layer.update({'layer_normalization': nn.LayerNorm(normalized_shape=out_feat, eps=1e-05,
                                                                  elementwise_affine=True)})

            if self.activation == 'leaky_relu':
                layer.update({'activation': nn.LeakyReLU(inplace=True)})
            elif self.activation == 'elu':
                layer.update({'activation': nn.ELU(inplace=True)})
            elif self.activation == 'selu':
                layer.update({'activation': nn.SELU(inplace=True)})
            elif self.activation == 'relu':
                layer.update({'activation': nn.ReLU(inplace=True)})
            elif self.activation == 'softplus':
                layer.update({'activation': nn.Softplus()})
            else:
                print('Inserted activation not available.')
        else:  # Normalization after Non-Linearity
            if self.activation == 'leaky_relu':
                layer.update({'activation': nn.LeakyReLU(inplace=True)})
            elif self.activation == 'elu':
                layer.update({'activation': nn.ELU(inplace=True)})
            elif self.activation == 'selu':
                layer.update({'activation': nn.SELU(inplace=True)})
            elif self.activation == 'relu':
                layer.update({'activation': nn.ReLU(inplace=True)})
            elif self.activation == 'softplus':
                layer.update({'activation': nn.Softplus()})
            else:
                print('Inserted activation not available.')

            if self.normalization == 'batch':
                layer.update(
                    {'batch_normalization': nn.BatchNorm1d(num_features=out_feat, eps=1e-05, momentum=0.1,
                                                           affine=True, track_running_stats=True)})
            elif self.normalization == 'layer':
                layer.update({'layer_normalization': nn.LayerNorm(normalized_shape=out_feat, eps=1e-05,
                                                                  elementwise_affine=True)})

        return layer

    def _create_model(self):
        """
        Helper function to create the model stacked into a nn.Sequential model.
        """
        modules = {}
        for i in range(len(self.layers) - 1):
            if i == len(self.layers) - 2:
                modules.update({"linear_{}".format(i): nn.Linear(in_features=self.layers[i],
                                                                 out_features=self.layers[i + 1])})
            else:
                current_dict_block = self._block(in_feat=self.layers[i], out_feat=self.layers[i + 1])
                for key, val in current_dict_block.items():
                    modules.update({
                        "{}_{}".format(key, i): val
                    })
        if self.use_tanh:
            modules.update({"output_activation": nn.Tanh()})

        # convert python dictionary to OrderedDict
        modules = OrderedDict(modules)
        sequential = nn.Sequential(modules)
        return sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == '__main__':
    model = Neuraldecipher()
    print(model)
