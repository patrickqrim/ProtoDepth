import os, sys
import torch
sys.path.insert(0, os.path.join('external_src', 'depth_completion'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'voiced'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'voiced', 'src'))
from voiced_model import VOICEDModel as VOICED
from posenet_model import PoseNetModel as PoseNet
from outlier_removal import OutlierRemoval


class VOICEDModel(object):
    '''
    Arg(s):
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
                 network_modules=['depth', 'pose'],
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        self.network_modules = network_modules

        # Instantiate depth completion model
        self.model_depth = VOICED(
            encoder_type='vggnet11',
            input_channels_image=3,
            input_channels_depth=2,
            n_filters_encoder_image=[48, 96, 192, 384, 384],
            n_filters_encoder_depth=[16, 32, 64, 128, 128],
            n_filters_decoder=[256, 128, 128, 64, 0],
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            device=device)

        if 'pose' in network_modules:
            self.model_pose = PoseNet(
                encoder_type='posenet',
                rotation_parameterization='euler',
                weight_initializer='xavier_normal',
                activation_func='leaky_relu',
                device=device)
        else:
            self.model_pose = None

        self.outlier_removal = OutlierRemoval(
            kernel_size=7,
            threshold=1.5)

        # Move to device
        self.device = device
        self.to(self.device)
        self.eval()

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

        # Normalize image between [0, 1]
        if torch.max(image) > 1.0:
            image = image / 255.0

        return image, sparse_depth, validity_map


    def forward_depth_encoder(self, image, sparse_depth, validity_map, intrinsics, return_all_outputs=False):
        '''
        Forwards stereo pair through the encoder

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
                if set, then return list of all outputs
        Returns:
            torch.Tensor[float32] : N x C x H x W latent representation
            list[torch.Tensor[float32]] : list of skip connections
            tuple : shape of the input
        '''
        
        image, sparse_depth, validity_map = self.transform_inputs(
            image=image,
            sparse_depth=sparse_depth,
            validity_map=validity_map)

        latent, skips, shape = self.model_depth.forward_encoder(
            image=image,
            sparse_depth=sparse_depth,
            validity_map=validity_map)

        return latent, skips, shape


    def forward_depth_decoder(self, latent, skips, shape, return_all_outputs=False):
        '''
        Forwards latent representation through the decoder

        Arg(s):
            latent : torch.Tensor[float32]
                N x C x H x W latent representation
            skips : list[torch.Tensor[float32]]
                list of skip connections
            shape : tuple
                shape of the input
            return_all_outputs : bool
                if set, then return list of all outputs
        Returns:   
            torch.Tensor[float32] : N x 1 x H x W output depth
        '''

        output_depth = self.model_depth.forward_decoder(
            latent=latent,
            skips=skips,
            shape=shape)

        if return_all_outputs:
            output_depth = [output_depth]

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
            image / 255.0 if torch.max(image) > 1.0 else image for image in [image0, image1]
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
        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm} + w_{pc}l_{pc}

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at time step t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at time step t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W image at time step t+1
            output_depth0 : list[torch.Tensor[float32]]
                list of N x 1 x H x W output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth at time t
            validity_map_depth0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth at time t
            validity_map_image0 : torch.Tensor[float32]
                N x 1 x H x W validity map of image at time t
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t-1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t+1
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_smoothness : float
                weight of local smoothness term
            w_pose : float
                weight of pose consistency term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        # Check if loss weighting was passed in, if not then use default weighting
        w_color = w_losses['w_color'] if 'w_color' in w_losses else 0.20
        w_structure = w_losses['w_structure'] if 'w_structure' in w_losses else 0.80
        w_sparse_depth = w_losses['w_sparse_depth'] if 'w_sparse_depth' in w_losses else 0.20
        w_smoothness = w_losses['w_smoothness'] if 'w_smoothness' in w_losses else 0.01
        w_pose = w_losses['w_pose'] if 'w_pose' in w_losses else 0.10

        # Unwrap from list
        output_depth0 = output_depth0[0]
        validity_map_image0 = validity_map_image0[0] if validity_map_image0 is not None else None

        # Remove outlier points and update sparse depth and validity map
        filtered_sparse_depth0, \
            filtered_validity_map_depth0 = self.outlier_removal.remove_outliers(
                sparse_depth=sparse_depth0,
                validity_map=validity_map_depth0)

        # Normalize images between [0, 1]
        image0, image1, image2 = [
            image / 255.0 for image in [image0, image1, image2] if torch.max(image) > 1.0
        ]

        # Compute pose from 1 to 0 and 2 to 0 for pose consistency loss
        pose1to0 = self.forward_pose(image1, image0)
        pose2to0 = self.forward_pose(image2, image0)

        loss, loss_info = self.model_depth.compute_loss(
            image0=image0,
            image1=image1,
            image2=image2,
            output_depth0=output_depth0,
            sparse_depth0=filtered_sparse_depth0,
            validity_map0=filtered_validity_map_depth0,
            validity_map_image0=validity_map_image0,
            intrinsics=intrinsics,
            pose0to1=pose0to1,
            pose0to2=pose0to2,
            pose1to0=pose1to0,
            pose2to0=pose2to0,
            w_color=w_color,
            w_structure=w_structure,
            w_sparse_depth=w_sparse_depth,
            w_smoothness=w_smoothness,
            w_pose=w_pose)

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = self.model_depth.parameters()

        if 'pose' in self.network_modules:
            parameters = parameters + self.model_pose.parameters()

        return parameters

    def parameters_depth(self):
        '''
        Fetches model parameters for depth network modules

        Returns:
            list[torch.Tensor[float32]] : list of model parameters for depth network modules
        '''

        return list(self.model_depth.parameters())

    def parameters_pose(self):
        '''
        Fetches model parameters for pose network modules

        Returns:
            list[torch.Tensor[float32]] : list of model parameters for pose network modules
        '''

        if 'pose' in self.network_modules:
            return list(self.model_pose.parameters())
        else:
            raise ValueError('Unsupported pose network architecture: {}'.format(self.network_modules))

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model_depth.train()

        if 'pose' in self.network_modules:
            self.model_pose.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model_depth.eval()

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

        self.model_depth.to(device)

        if 'pose' in self.network_modules:
            self.model_pose.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        # KBNet and PoseNet already call data_parallel() in constructor
        self.model_depth.data_parallel()

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

        train_step, optimizer_depth = self.model_depth.restore_model(
            checkpoint_path=model_depth_restore_path,
            optimizer=optimizer_depth)

        if 'pose' in self.network_modules and model_pose_restore_path is not None:
            _, optimizer_pose = self.model_pose.restore_model(
                model_pose_restore_path,
                optimizer_pose)

        if optimizer_depth is None and optimizer_pose is None:
            return train_step
        else:
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

        self.model_depth.save_model(
            checkpoint_path=model_depth_checkpoint_path,
            step=step,
            optimizer=optimizer_depth)

        if 'pose' in self.network_modules and model_pose_checkpoint_path is not None:
            self.model_pose.save_model(
                checkpoint_path=model_pose_checkpoint_path,
                step=step,
                optimizer=optimizer_pose)
