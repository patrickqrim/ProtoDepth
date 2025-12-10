import os, sys
import torch
sys.path.insert(0, os.path.join('external_src', 'depth_completion'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'scaffnet'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'scaffnet', 'src'))
from scaffnet_model import ScaffNetModel as ScaffNet


class ScaffNetModel(object):
    '''
    Class for interfacing with ScaffNet model

    Arg(s):
        dataset_name : str
            model for a given dataset
        network_modules : list[str]
            network modules to build for model
        min_predict_depth : float
            minimum value of predicted depth
        max_predict_depth : flaot
            maximum value of predicted depth
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 dataset_name='vkitti',
                 network_modules=['depth'],
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        encoder_type = ['batch_norm']
        decoder_type = ['multi-scale', 'batch_norm']

        if 'resnet18' in network_modules:
            encoder_type.append('resnet18')
        elif 'vggnet08' in network_modules:
            encoder_type.append('vggnet08')
        else:
            encoder_type.append('vggnet08')

        if 'spatial_pyramid_pool' in network_modules:
            encoder_type.append('spatial_pyramid_pool')

        if 'uncertainty' in network_modules:
            decoder_type.append('uncertainty')

        # Instantiate depth completion model
        if dataset_name == 'kitti' or dataset_name == 'vkitti':
            max_pool_sizes_spatial_pyramid_pool = [13, 17, 19, 21, 25]
            n_convolution_spatial_pyramid_pool = 3
            n_filter_spatial_pyramid_pool = 8
            n_filters_encoder = [16, 32, 64, 128, 256]
            n_filters_decoder = [256, 128, 128, 64, 32]
            min_predict_depth = min_predict_depth
            max_predict_depth = max_predict_depth
        elif dataset_name == 'void' or dataset_name == 'scenenet' or dataset_name == 'nyu_v2':
            max_pool_sizes_spatial_pyramid_pool = [13, 17, 19, 21, 25]
            n_convolution_spatial_pyramid_pool = 3
            n_filter_spatial_pyramid_pool = 8
            n_filters_encoder = [16, 32, 64, 128, 256]
            n_filters_decoder = [256, 128, 128, 64, 32]
            min_predict_depth = min_predict_depth
            max_predict_depth = max_predict_depth
        else:
            raise ValueError('Unsupported dataset settings: {}'.format(dataset_name))

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        # Build ScaffNet
        self.model = ScaffNet(
            max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
            n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
            n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
            encoder_type=encoder_type,
            n_filters_encoder=n_filters_encoder,
            decoder_type=decoder_type,
            n_filters_decoder=n_filters_decoder,
            n_output_resolution=1,
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            device=device)

        # Move to device
        self.device = device
        self.to(self.device)

        # Freeze ScaffNet
        if 'freeze_depth' in network_modules:
            self.model.freeze(
                module_names=['spatial_pyramid_pool', 'encoder', 'decoder_depth'])
        elif 'freeze_all' in network_modules:
            self.model.freeze(
                module_names=['all'])

    def forward_depth(self, image, sparse_depth, validity_map, intrinsics, return_all_outputs=False):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            validity_map : torch.Tensor[float32]
                N x 1 x H x W valid locations of projected sparse point cloud
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
            return_all_outputs : bool
                if set, return all outputs
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''

        # Forward through ScaffNet
        output_depth = self.model.forward(sparse_depth)

        if return_all_outputs:
            if 'uncertainty' in self.model.decoder_type:
                return torch.chunk(output_depth, chunks=2, dim=1)

            return [output_depth]
        else:
            if 'uncertainty' in self.model.decoder_type:
                output_depth = output_depth[:, 0:1, :, :]

            return output_depth

    def transform_inputs(self, image, sparse_depth):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
        Returns:
            torch.Tensor[float32] : N x 3 x H x W image
            torch.Tensor[float32] : N x 1 x H x W sparse depth map
        '''

        return image, sparse_depth

    def compute_loss(self,
                     target_depth,
                     output_depth,
                     w_supervised=1.00):
        '''
        Computes loss function

        Arg(s):
            target_depth : torch.Tensor[float32]
                N x 1 x H x W groundtruth target depth
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth
            w_supervised : float
                weight of supervised loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : loss related infor
        '''

        # Separate uncertainty from depth
        if 'uncertainty' in self.model.decoder_type:
            output_uncertainty = output_depth[-1]
        else:
            output_uncertainty = None

        # Compute loss function
        loss, loss_info = self.model.compute_loss(
            loss_func='supervised_l1_normalized',
            target_depth=target_depth,
            output_depths=output_depth,
            output_uncertainties=output_uncertainty,
            w_supervised=w_supervised)

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters()

    def parameters_depth(self):
        '''
        Returns the list of parameters in the model
        Because there is no pose network, it is a wrapper for parameters

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.parameters()

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.model.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model.data_parallel()

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optim
                optimizer
        '''

        _, optimizer = self.model.restore_model(
            checkpoint_path=restore_path,
            optimizer=optimizer)

        return optimizer

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        self.model.save_model(
            checkpoint_path=checkpoint_path,
            step=step,
            optimizer=optimizer)
