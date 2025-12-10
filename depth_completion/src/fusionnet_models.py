import os, sys
import torch
sys.path.insert(0, os.path.join('external_src', 'depth_completion'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'scaffnet'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'scaffnet', 'src'))
from net_utils import OutlierRemoval
from fusionnet_model import FusionNetModel as FusionNet
from posenet_model import PoseNetModel as PoseNet
from scaffnet_models import ScaffNetModel


class FusionNetModel(object):
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
                 network_modules=['depth', 'pose'],
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        self.network_modules = network_modules

        if dataset_name == 'kitti' or dataset_name == 'vkitti':
            min_predict_depth_scaffnet = 1.5
            max_predict_depth_scaffnet = 80.0
        elif dataset_name == 'void' or dataset_name == 'scenenet' or dataset_name == 'nyu_v2':
            min_predict_depth_scaffnet = 0.1
            max_predict_depth_scaffnet = 10.0
        else:
            raise ValueError('Unsupported dataset settings: {}'.format(dataset_name))

        encoder_type_fusionnet = ['vggnet08']
        n_filters_encoder_image_fusionnet = [48, 96, 192, 384, 384]
        n_filters_encoder_depth_fusionnet = [16, 32, 64, 128, 128]
        decoder_type_fusionnet = ['multi-scale']
        n_filters_decoder_fusionnet = [256, 128, 128, 64, 32]
        scale_match_method_fusionnet = 'local_scale'
        scale_match_kernel_size_fusionnet = 5
        min_predict_depth_fusionnet = min_predict_depth
        max_predict_depth_fusionnet = max_predict_depth
        min_multiplier_depth_fusionnet = 0.25
        max_multiplier_depth_fusionnet = 4.00
        min_residual_depth_fusionnet = -1000.0
        max_residual_depth_fusionnet = 1000.0

        # Build ScaffNet
        network_modules_scaffnet = network_modules + ['freeze_all']

        self.scaffnet_model = ScaffNetModel(
            dataset_name=dataset_name,
            network_modules=network_modules_scaffnet,
            min_predict_depth=min_predict_depth_scaffnet,
            max_predict_depth=max_predict_depth_scaffnet,
            device=device)

        # Set Scaffnet to evalulation mode
        self.scaffnet_model.eval()

        # Build FusionNet
        self.fusionnet_model = FusionNet(
            encoder_type=encoder_type_fusionnet,
            n_filters_encoder_image=n_filters_encoder_image_fusionnet,
            n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
            decoder_type=decoder_type_fusionnet,
            n_filters_decoder=n_filters_decoder_fusionnet,
            scale_match_method=scale_match_method_fusionnet,
            scale_match_kernel_size=scale_match_kernel_size_fusionnet,
            min_predict_depth=min_predict_depth_fusionnet,
            max_predict_depth=max_predict_depth_fusionnet,
            min_multiplier_depth=min_multiplier_depth_fusionnet,
            max_multiplier_depth=max_multiplier_depth_fusionnet,
            min_residual_depth=min_residual_depth_fusionnet,
            max_residual_depth=max_residual_depth_fusionnet,
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            device=device)

        # Build outlier removal
        self.outlier_removal = OutlierRemoval(
            kernel_size=7,
            threshold=1.5)

        # Build pose network
        if 'pose' in network_modules:
            self.model_pose = PoseNet(
                encoder_type='posenet',
                rotation_parameterization='axis',
                weight_initializer='xavier_normal',
                activation_func='relu',
                device=device)
        else:
            self.model_pose = None

        # TokenCDC: Must store input_depth as a global variable for FusionNet only
        self.input_depth = None

        # Move to device
        self.device = device
        self.to(self.device)


    def transform_inputs(self, image, sparse_depth, validity_map):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            validity_map : torch.Tensor[float32]
                N x 1 x H x W valid locations of projected sparse point cloud
        Returns:
            torch.Tensor[float32] : N x 3 x H x W image
            torch.Tensor[float32] : N x 1 x H x W sparse depth map
            torch.Tensor[float32] : N x 1 x H x W validity map
        '''

        # Remove outlier points and update sparse depth and validity map
        filtered_sparse_depth, \
            filtered_validity_map = self.outlier_removal.remove_outliers(
                sparse_depth=sparse_depth,
                validity_map=validity_map)

        # Normalize image between [0, 1]
        if torch.max(image) > 1.0:
            image = image / 255.0

        return image, filtered_sparse_depth, filtered_validity_map

    def forward_depth_encoder(self, image, sparse_depth, validity_map, intrinsics, return_all_outputs=False):
        '''
        Forwards inputs through the encoder

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
            torch.Tensor[float32] : N x C x H x W latent representation
            list[torch.Tensor[float32]] : list of skip connections
            tuple[int] : shape of latent representation
        '''

        # Forward through ScaffNet
        input_depth = self.scaffnet_model.forward_depth(
            image,
            sparse_depth,
            validity_map,
            intrinsics,
            return_all_outputs=False)
        self.input_depth = input_depth

        if 'uncertainty' in self.scaffnet_model.decoder_type:
            if return_all_outputs:
                input_depth = input_depth[0]

        image, \
            filtered_sparse_depth, \
            filtered_validity_map = self.transform_inputs(
                image=image,
                sparse_depth=sparse_depth,
                validity_map=validity_map)

        # Forward through FusionNet Encoder
        latent, skips, shape = self.fusionnet_model.forward_encoder(
            image=image,
            input_depth=input_depth,
            sparse_depth=filtered_sparse_depth)
            
        return latent, skips, shape


    def forward_depth_decoder(self, latent, skips, shape, return_all_outputs=False):
        '''
        Forwards latent representation through the decoder

        Arg(s):
            latent : torch.Tensor[float32]
                N x C x H x W latent representation
            skips : list[torch.Tensor[float32]]
                list of skip connections
            shape : tuple[int]
                shape of latent representation
            return_all_outputs : bool
                if set, return all outputs
        Returns:
            torch.Tensor[float32] : N x 1 x H x W output depth
            torch.Tensor[float32] : N x 1 x H x W input depth
        '''

        output_depth = self.fusionnet_model.forward_decoder(
            latent=latent,
            skips=skips,
            shape=shape)
        input_depth = self.input_depth

        if return_all_outputs:
            return [output_depth, input_depth]
        else:
            return output_depth

    def forward_pose(self, image0, image1):
        '''
        Forwards a pair of images through the network to output pose from time 0 to 1

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W tensor
            image1 : torch.Tensor[float32]
                N x C x H x W tensor
        Returns:
            torch.Tensor[float32] : N x 4 x 4  pose matrix
        '''

        assert self.model_pose is not None

        # Normalize image between [0, 1]
        image0, image1 = [
            image / 255.0 for image in [image0, image1] if torch.max(image) > 1.0
        ]

        return self.model_pose.forward(image0, image1)

    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     sparse_depth0,
                     validity_map_depth0,
                     validity_map_image0,
                     intrinsics,
                     pose0to1,
                     pose0to2,
                     w_losses):
        '''
        Computes loss function

        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm} + w_{tp}l_{tp}

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W time t image
            image1 : torch.Tensor[float32]
                N x 3 x H x W time t-1 image
            image2 : torch.Tensor[float32]
                N x 3 x H x W time t+1 image
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth and input depth from ScaffNet for time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth for time t
            validity_map_depth0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth at time t
            validity_map_image0 : torch.Tensor[float32]
                N x 1 x H x W validity map of image at time t
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from time t to t-1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from time t to t+1
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_smoothness : float
                weight of local smoothness term
            w_prior_depth : float
                weight of prior depth consistency term
            threshold_prior_depth : float
                threshold to start using prior depth term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        # Check if loss weighting was passed in, if not then use default weighting
        w_color = w_losses['w_color'] if 'w_color' in w_losses else 0.20
        w_structure = w_losses['w_structure'] if 'w_structure' in w_losses else 0.80
        w_sparse_depth = w_losses['w_sparse_depth'] if 'w_sparse_depth' in w_losses else 0.20
        w_smoothness = w_losses['w_smoothness'] if 'w_smoothness' in w_losses else 0.01
        w_prior_depth = w_losses['w_prior_depth'] if 'w_prior_depth' in w_losses else 0.10
        threshold_prior_depth = w_losses['threshold_prior_depth'] if 'threshold_prior_depth' in w_losses else 0.30

        # Unwrap from list
        input_depth0 = output_depth0[1]
        output_depth0 = output_depth0[0]
        validity_map_image0 = validity_map_image0[0]if validity_map_image0 is not None else None

        # Remove outlier points and update sparse depth and validity map
        filtered_sparse_depth0, \
            filtered_validity_map_depth0 = self.outlier_removal.remove_outliers(
                sparse_depth=sparse_depth0,
                validity_map=validity_map_depth0)

        # Normalize images between [0, 1]
        image0, image1, image2 = [
            image / 255.0 for image in [image0, image1, image2] if torch.max(image) > 1.0
        ]

        # Compute loss function
        loss, loss_info = self.fusionnet_model.compute_loss_unsupervised(
            output_depth0=output_depth0,
            sparse_depth0=filtered_sparse_depth0,
            validity_map0=filtered_validity_map_depth0,
            validity_map_image0=validity_map_image0,
            input_depth0=input_depth0,
            image0=image0,
            image1=image1,
            image2=image2,
            pose0to1=pose0to1,
            pose0to2=pose0to2,
            intrinsics=intrinsics,
            w_color=w_color,
            w_structure=w_structure,
            w_sparse_depth=w_sparse_depth,
            w_smoothness=w_smoothness,
            w_prior_depth=w_prior_depth,
            threshold_prior_depth=threshold_prior_depth)

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = list(self.fusionnet_model.parameters())

        if 'pose' in self.network_modules:
            parameters = parameters + list(self.model_pose.parameters())

        return parameters

    def parameters_depth(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.fusionnet_model.parameters()

    def parameters_pose(self):
        '''
        Fetches model parameters for pose network modules

        Returns:
            list[torch.Tensor[float32]] : list of model parameters for pose network modules
        '''

        if 'pose' in self.network_modules:
            return self.model_pose.parameters()
        else:
            raise ValueError('Unsupported pose network architecture: {}'.format(self.network_modules))

    def train(self):
        '''
        Sets model to training mode
        '''

        # ScaffNet is frozen
        self.fusionnet_model.train()

        if 'pose' in self.network_modules:
            self.model_pose.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.scaffnet_model.eval()
        self.fusionnet_model.eval()

        if 'pose' in self.network_modules:
            self.model_pose.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.scaffnet_model.to(device)
        self.fusionnet_model.to(device)

        if 'pose' in self.network_modules:
            self.model_pose.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.scaffnet_model.data_parallel()
        self.fusionnet_model.data_parallel()

        if 'pose' in self.network_modules:
            self.model_pose.data_parallel()

    def restore_model(self,
                      model_depth_restore_path,
                      model_pose_restore_path=None,
                      optimizer_depth=None,
                      optimizer_pose=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            model_depth_restore_path : str
                path to model weights for depth network
            model_pose_restore_path : str
                path to model weights for pose network
            optimizer_depth : torch.optim
                optimizer for depth network
            optimizer_pose : torch.optim
                optimizer for depth network
        '''

        train_step, optimizer_depth = self.fusionnet_model.restore_model(
            checkpoint_path=model_depth_restore_path,
            optimizer=optimizer_depth,
            scaffnet_model=self.scaffnet_model.model)

        if 'pose' in self.network_modules and model_pose_restore_path is not None:
            _, optimizer_pose = self.model_pose.restore_model(
                model_pose_restore_path,
                optimizer_pose)

        return train_step, optimizer_depth, optimizer_pose

    def save_model(self,
                   model_depth_checkpoint_path,
                   step,
                   optimizer_depth,
                   model_pose_checkpoint_path=None,
                   optimizer_pose=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            model_depth_checkpoint_path : str
                path to save checkpoint for depth network
            step : int
                current training step
            optimizer_depth : torch.optim
                optimizer for depth network
            model_pose_checkpoint_path : str
                path to save checkpoint for pose network
            optimizer_pose : torch.optim
                optimizer for pose network
        '''

        self.fusionnet_model.save_model(
            checkpoint_path=model_depth_checkpoint_path,
            step=step,
            optimizer=optimizer_depth,
            scaffnet_model=self.scaffnet_model.model)

        if 'pose' in self.network_modules and model_pose_checkpoint_path is not None:
            self.model_pose.save_model(
                checkpoint_path=model_pose_checkpoint_path,
                step=step,
                optimizer=optimizer_pose)
