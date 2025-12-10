import os, torch, torchvision
import torch.nn.functional as F
from utils.src import log_utils, net_utils
from continual_learning_losses import agnostic_loss


class DepthCompletionModel(object):
    '''
    Wrapper class for all external depth completion models

    Arg(s):
        model_name : str
            depth completion model to use
        network_modules : list[str]
            network modules to build for model
        min_predict_depth : float
            minimum depth to predict
        max_predict_depth : float
            maximum depth to predict
        image_pool_size : int
            size of image pool for TokenCDC
        depth_pool_size : int
            size of depth pool for TokenCDC
        frozen : bool
            for TokenCDC, freeze the model if True
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 model_name,
                 network_modules,
                 min_predict_depth,
                 max_predict_depth,
                 image_pool_size,
                 depth_pool_size,
                 unfrozen=False,
                 device=torch.device('cuda')):

        self.model_name = model_name
        self.network_modules = network_modules
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.image_pool_size = image_pool_size
        self.depth_pool_size = depth_pool_size
        self.frozen = not unfrozen
        self.device = device

        # Parse dataset name
        if 'kitti' in model_name:
            dataset_name = 'kitti'
        elif 'vkitti' in model_name:
            dataset_name = 'vkitti'
        elif 'void' in model_name:
            dataset_name = 'void'
        elif 'scenenet' in model_name:
            dataset_name = 'scenenet'
        elif 'nyu_v2' in model_name:
            dataset_name = 'nyu_v2'
        else:
            dataset_name = 'kitti'

        if 'kbnet' in model_name:
            from kbnet_models import KBNetModel

            self.model = KBNetModel(
                dataset_name=dataset_name,
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'scaffnet' in model_name:
            from scaffnet_models import ScaffNetModel

            self.model = ScaffNetModel(
                dataset_name=dataset_name,
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'fusionnet' in model_name:
            from fusionnet_models import FusionNetModel

            self.model = FusionNetModel(
                dataset_name=dataset_name,
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'voiced' in model_name:
            from voiced_models import VOICEDModel

            self.model = VOICEDModel(
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(model_name))

        # Freeze depth model parameters if unfrozen=False
        if self.frozen:
            for param in self.model.parameters_depth():
                param.requires_grad = False

        # Initialize continual learning model
        from continual_learning_model import ContinualLearningModel
        self.model_cl = ContinualLearningModel(
            image_pool_size=image_pool_size,
            depth_pool_size=depth_pool_size,
            device=device)

    def _get_model_cl(self):
        '''
        To be compatible with DataParallel
        '''
        return self.model_cl.module if isinstance(self.model_cl, torch.nn.DataParallel) else self.model_cl


    def forward_depth(self,
                      image,
                      sparse_depth,
                      validity_map,
                      dataset_uid,
                      intrinsics=None, 
                      domain_agnostic_eval=False,
                      return_all_outputs=False):
        '''
        Forwards stereo pair through network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            validity_map : torch.Tensor[float32]
                N x 1 x H x W valid locations of projected sparse point cloud
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
            dataset_uid : str
                unique id of dataset
            return_all_outputs : bool
                if set, then return list of N x 1 x H x W depth maps else a single N x 1 x H x W depth map
        Returns:
            list[torch.Tensor[float32]] : a single or list of N x 1 x H x W outputs
        '''

        # Encoder Forward Pass
        latent, skips, shape = self.model.forward_depth_encoder(
            image, 
            sparse_depth,
            validity_map,
            intrinsics)

        if 'control' not in self.network_modules:
            # Selector query
            selector_query = F.adaptive_avg_pool2d(latent, (1, 1)).view(latent.shape[0], -1)
            dataset_selectors = self._get_model_cl().dataset_selectors
            if domain_agnostic_eval:
                with torch.no_grad():
                    selector_query_norm = F.normalize(selector_query, p=2, dim=1)
                    selector_key_matrix = torch.cat(dataset_selectors, dim=1)
                    selector_key_matrix_norm = F.normalize(selector_key_matrix, p=2, dim=0)
                    cosine_sim = selector_query_norm @ selector_key_matrix_norm
                    selector_key_idx = torch.argmax(cosine_sim.mean(dim=0)).item()
                    dataset_uid = self._get_model_cl().dataset_uids[selector_key_idx]

            # Get key and token pools
            dims = skips[1][0].shape[1], skips[1][1].shape[1], \
                    skips[2][0].shape[1], skips[2][1].shape[1], skips[3][0].shape[1], skips[3][1].shape[1], latent.shape[1]
            curr_i2_key_pool, curr_i2_token_pool, curr_i2_linear, curr_d2_key_pool, curr_d2_token_pool, curr_d2_linear, \
            curr_i3_key_pool, curr_i3_token_pool, curr_i3_linear, curr_d3_key_pool, curr_d3_token_pool, curr_d3_linear, \
            curr_i4_key_pool, curr_i4_token_pool, curr_i4_linear, curr_d4_key_pool, curr_d4_token_pool, curr_d4_linear, \
            curr_latent_key_pool, curr_latent_token_pool, curr_latent_linear = \
                self._get_model_cl().get_key_token_pool(dataset_uid, dims)
            selector_key_idx = self._get_model_cl().get_selector_key_idx(dataset_uid)

            # Skip connection 1
            i1_skip_with_tokens, d1_skip_with_tokens = skips[0][0], skips[0][1]

            # Skip connection 2
            i2_skip, d2_skip = skips[1][0], skips[1][1]
            i2_N, i2_C, i2_H, i2_W = i2_skip.shape
            d2_N, d2_C, d2_H, d2_W = d2_skip.shape
            assert i2_N == d2_N and i2_H == d2_H and i2_W == d2_W, "Image/depth must have the same non-channel dims!"
            i2_queries = i2_skip.permute(0, 2, 3, 1).view(i2_N, i2_H*i2_W, i2_C)
            d2_queries = d2_skip.permute(0, 2, 3, 1).view(d2_N, d2_H*d2_W, d2_C)
            # Compute tokens using attention
            i2_keys = torch.matmul(curr_i2_key_pool, curr_i2_token_pool.detach().clone().transpose(-2, -1))
            i2_scores = torch.matmul(i2_queries, i2_keys) / torch.sqrt(torch.tensor(i2_C, device=self.device, dtype=torch.float32))
            i2_scores = F.softmax(i2_scores, dim=-1)
            d2_keys = torch.matmul(curr_d2_key_pool, curr_d2_token_pool.detach().clone().transpose(-2, -1))
            d2_scores = torch.matmul(d2_queries, d2_keys) / torch.sqrt(torch.tensor(d2_C, device=self.device, dtype=torch.float32))
            d2_scores = F.softmax(d2_scores, dim=-1)
            i2_tokens = torch.matmul(i2_scores, curr_i2_token_pool).view(i2_N, i2_H, i2_W, i2_C).permute(0, 3, 1, 2)
            d2_tokens = torch.matmul(d2_scores, curr_d2_token_pool).view(d2_N, d2_H, d2_W, d2_C).permute(0, 3, 1, 2)
            # Apply linear layer to skips
            i2_skip = i2_skip * curr_i2_linear
            d2_skip = d2_skip * curr_d2_linear
            # Concatenate tokens to skips
            i2_skip_with_tokens = i2_skip + i2_tokens
            d2_skip_with_tokens = d2_skip + d2_tokens

            # Skip connection 3
            i3_skip, d3_skip = skips[2][0], skips[2][1]
            i3_N, i3_C, i3_H, i3_W = i3_skip.shape
            d3_N, d3_C, d3_H, d3_W = d3_skip.shape
            assert i3_N == d3_N and i3_H == d3_H and i3_W == d3_W, "Image/depth must have the same non-channel dims!"
            i3_queries = i3_skip.permute(0, 2, 3, 1).view(i3_N, i3_H*i3_W, i3_C)
            d3_queries = d3_skip.permute(0, 2, 3, 1).view(d3_N, d3_H*d3_W, d3_C)
            # Compute tokens using attention
            i3_keys = torch.matmul(curr_i3_key_pool, curr_i3_token_pool.detach().clone().transpose(-2, -1))
            i3_scores = torch.matmul(i3_queries, i3_keys) / torch.sqrt(torch.tensor(i3_C, device=self.device, dtype=torch.float32))
            i3_scores = F.softmax(i3_scores, dim=-1)
            d3_keys = torch.matmul(curr_d3_key_pool, curr_d3_token_pool.detach().clone().transpose(-2, -1))
            d3_scores = torch.matmul(d3_queries, d3_keys) / torch.sqrt(torch.tensor(d3_C, device=self.device, dtype=torch.float32))
            d3_scores = F.softmax(d3_scores, dim=-1)
            i3_tokens = torch.matmul(i3_scores, curr_i3_token_pool).view(i3_N, i3_H, i3_W, i3_C).permute(0, 3, 1, 2)
            d3_tokens = torch.matmul(d3_scores, curr_d3_token_pool).view(d3_N, d3_H, d3_W, d3_C).permute(0, 3, 1, 2)
            # Apply linear layer to skips
            i3_skip = i3_skip * curr_i3_linear
            d3_skip = d3_skip * curr_d3_linear
            # Concatenate tokens to skips
            i3_skip_with_tokens = i3_skip + i3_tokens
            d3_skip_with_tokens = d3_skip + d3_tokens

            # Skip connection 4
            i4_skip, d4_skip = skips[3][0], skips[3][1]
            i4_N, i4_C, i4_H, i4_W = i4_skip.shape
            d4_N, d4_C, d4_H, d4_W = d4_skip.shape
            assert i4_N == d4_N and i4_H == d4_H and i4_W == d4_W, "Image/depth must have the same non-channel dims!"
            i4_queries = i4_skip.permute(0, 2, 3, 1).view(i4_N, i4_H*i4_W, i4_C)
            d4_queries = d4_skip.permute(0, 2, 3, 1).view(d4_N, d4_H*d4_W, d4_C)
            # Compute tokens using attention
            i4_keys = torch.matmul(curr_i4_key_pool, curr_i4_token_pool.detach().clone().transpose(-2, -1))
            i4_scores = torch.matmul(i4_queries, i4_keys) / torch.sqrt(torch.tensor(i4_C, device=self.device, dtype=torch.float32))
            i4_scores = F.softmax(i4_scores, dim=-1)
            d4_keys = torch.matmul(curr_d4_key_pool, curr_d4_token_pool.detach().clone().transpose(-2, -1))
            d4_scores = torch.matmul(d4_queries, d4_keys) / torch.sqrt(torch.tensor(d4_C, device=self.device, dtype=torch.float32))
            d4_scores = F.softmax(d4_scores, dim=-1)
            i4_tokens = torch.matmul(i4_scores, curr_i4_token_pool).view(i4_N, i4_H, i4_W, i4_C).permute(0, 3, 1, 2)
            d4_tokens = torch.matmul(d4_scores, curr_d4_token_pool).view(d4_N, d4_H, d4_W, d4_C).permute(0, 3, 1, 2)
            # Apply linear layer to skips
            i4_skip = i4_skip * curr_i4_linear
            d4_skip = d4_skip * curr_d4_linear
            # Concatenate tokens to skips
            i4_skip_with_tokens = i4_skip + i4_tokens
            d4_skip_with_tokens = d4_skip + d4_tokens

            # Latent space
            latent_N, latent_C, latent_H, latent_W = latent.shape
            latent_queries = latent.permute(0, 2, 3, 1).view(latent_N, latent_H*latent_W, latent_C)
            # Compute tokens using attention
            latent_keys = torch.matmul(curr_latent_key_pool, curr_latent_token_pool.detach().clone().transpose(-2, -1))
            latent_scores = torch.matmul(latent_queries, latent_keys) / torch.sqrt(torch.tensor(latent_C, device=self.device, dtype=torch.float32))
            latent_scores = F.softmax(latent_scores, dim=-1)
            latent_tokens = torch.matmul(latent_scores, curr_latent_token_pool).view(latent.shape[0], latent.shape[2], latent.shape[3], latent.shape[1]).permute(0, 3, 1, 2)
            # Apply linear layer to latent
            latent = latent * curr_latent_linear
            # Concatenate tokens to latent
            latent_with_tokens = latent + latent_tokens

            # Combine all skip connections
            skips_with_tokens = [torch.cat([i1_skip_with_tokens, d1_skip_with_tokens], dim=1), 
                                    torch.cat([i2_skip_with_tokens, d2_skip_with_tokens], dim=1),
                                    torch.cat([i3_skip_with_tokens, d3_skip_with_tokens], dim=1),
                                    torch.cat([i4_skip_with_tokens, d4_skip_with_tokens], dim=1)]
        else:
            latent_with_tokens = latent
            skips_with_tokens = [torch.cat([skips[0][0], skips[0][1]], dim=1), 
                                    torch.cat([skips[1][0], skips[1][1]], dim=1),
                                    torch.cat([skips[2][0], skips[2][1]], dim=1),
                                    torch.cat([skips[3][0], skips[3][1]], dim=1)]
            selector_query = F.adaptive_avg_pool2d(latent, (1, 1)).view(latent.shape[0], -1)
            selector_key_idx = None
            dataset_selectors = None

        # Decoder Forward Pass
        output = self.model.forward_depth_decoder(
                    latent_with_tokens,
                    skips_with_tokens,
                    shape,
                    return_all_outputs)

        return output, selector_query, selector_key_idx, dataset_selectors


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

        return self.model.forward_pose(image0, image1)


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
                     queries=None,
                     key_idx=None,
                     key_list=None,
                     domain_agnostic=False,
                     ground_truth0=None,
                     supervision_type='unsupervised',
                     w_losses={}):
        '''
        Call model's compute loss function

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
            queries : torch.Tensor[float32]
                N x latent_dim queries (FROZEN)
            key_idx : torch.Tensor[int64]
                N key index
            key_list : torch.Tensor[float32]
                latent_dim x K keys (LEARNABLE)
            ground_truth0 : torch.Tensor[float32]
                N x 1 x H x W ground truth depth at time t
            supervision_type : str
                type of supervision for training
            w_losses : dict[str, float]
                dictionary of weights for each loss
        Returns:
            float : loss averaged over the batch
            dict[str, float] : loss info
        '''

        if supervision_type == 'supervised':
            loss, loss_info = self.model.compute_loss(
                target_depth=ground_truth0,
                output_depth=output_depth0)
        elif supervision_type == 'unsupervised':
            loss, loss_info = self.model.compute_loss(
                image0=image0,
                image1=image1,
                image2=image2,
                output_depth0=output_depth0,
                sparse_depth0=sparse_depth0,
                validity_map_depth0=validity_map_depth0,
                validity_map_image0=validity_map_image0,
                intrinsics=intrinsics,
                pose0to1=pose0to1,
                pose0to2=pose0to2,
                w_losses=w_losses)
        else:
            raise ValueError('Unsupported supervision type: {}'.format(supervision_type))

        # Loss between selector query/key and between keys in the selector key matrix
        if domain_agnostic:
            loss_agnostic = agnostic_loss(
                                queries=queries,
                                key_idx=key_idx,
                                key_list=key_list,
                                lambda_agnostic=w_losses['w_agnostic'],
                                lambda_kk=w_losses['w_kk'])
            loss += loss_agnostic
            loss_info['loss_agnostic'] = loss_agnostic

        return loss, loss_info


    def get_new_params(self):
        '''
        Returns the list of new parameters added to the model

        Returns:
            list[torch.Tensor[float32]] : list of new parameters
        '''
        return self._get_model_cl().get_new_params()


    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        return list(self.model.parameters()) + list(self.model_cl.parameters())

    def parameters_depth(self):
        '''
        Returns the list of parameters in the depthmodel

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        return list(self.model.parameters_depth())

    def parameters_pose(self):
        '''
        Returns the list of parameters in the pose model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        return self.model.parameters_pose()
    
    def parameters_cl(self):
        '''
        Returns the list of parameters in the continual learning model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        return self.model_cl.parameters()


    def train(self):
        '''
        Sets model to training mode
        '''
        self.model.train()
        self.model_cl.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''
        self.model.eval()
        self.model_cl.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''
        self.device = device
        self.model.to(device)
        self._get_model_cl().to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''
        self.model.data_parallel()
        self.model_cl = torch.nn.DataParallel(self.model_cl)


    def restore_model(self,
                      restore_paths,
                      optimizer_depth=None,
                      optimizer_pose=None,
                      optimizer_cl=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_paths : list[str]
                path to model weights, 1st for depth model and 2nd for pose model (if exists), 3rd for continual learning
            optimizers
                current optimizers
        Returns:
            int : training step
            torch.optimizer : optimizer for depth or None if no optimizer is passed in
            torch.optimizer : optimizer for pose or None if no optimizer is passed in
            torch.optimizer : optimizer for continual learning or None if no optimizer is passed in
        '''

        if 'kbnet' in self.model_name:
            train_step, optimizer_depth, optimizer_pose = self.model.restore_model(
                                                                model_depth_restore_path=restore_paths[0],
                                                                model_pose_restore_path=restore_paths[1] if len(restore_paths) > 1 else None,
                                                                optimizer_depth=optimizer_depth,
                                                                optimizer_pose=optimizer_pose)
        elif 'scaffnet' in self.model_name:
            train_step, optimizer_depth, optimizer_pose = self.model.restore_model(
                                                                restore_path=restore_paths[0],
                                                                optimizer=optimizer_depth)
        elif 'fusionnet' in self.model_name:
            if 'initialize_scaffnet' in self.network_modules:
                self.model.scaffnet_model.restore_model(
                    restore_path=restore_paths[0])
                train_step, optimizer_depth, optimizer_pose = 0, optimizer_depth, optimizer_pose
            else:
                train_step, optimizer_depth, optimizer_pose = self.model.restore_model(
                                                                model_depth_restore_path=restore_paths[0],
                                                                model_pose_restore_path=restore_paths[1] if len(restore_paths) > 1 else None,
                                                                optimizer_depth=optimizer_depth,
                                                                optimizer_pose=optimizer_pose)
        elif 'voiced' in self.model_name:
            train_step, optimizer_depth, optimizer_pose = self.model.restore_model(
                                                                model_depth_restore_path=restore_paths[0],
                                                                model_pose_restore_path=restore_paths[1] if len(restore_paths) > 1 else None,
                                                                optimizer_depth=optimizer_depth,
                                                                optimizer_pose=optimizer_pose)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(self.model_name))

        return train_step, optimizer_depth, optimizer_pose, optimizer_cl


    def save_model(self,
                   checkpoint_dirpath,
                   step,
                   optimizer_depth=None,
                   optimizer_pose=None,
                   optimizer_cl=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_dirpath : str
                path to save directory to save checkpoints
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        os.makedirs(checkpoint_dirpath, exist_ok=True)

        if 'kbnet' in self.model_name:
            self.model.save_model(
                os.path.join(checkpoint_dirpath, 'kbnet-{}.pth'.format(step)),
                step,
                optimizer_depth,
                model_pose_checkpoint_path=os.path.join(checkpoint_dirpath, 'posenet-{}.pth'.format(step)),
                optimizer_pose=optimizer_pose)
        elif 'scaffnet' in self.model_name:
            self.model.save_model(
                os.path.join(checkpoint_dirpath, 'scaffnet-{}.pth'.format(step)),
                step=step,
                optimizer=optimizer_depth)
        elif 'fusionnet' in self.model_name:
            self.model.save_model(
                os.path.join(checkpoint_dirpath, 'fusionnet-{}.pth'.format(step)),
                step,
                optimizer_depth,
                model_pose_checkpoint_path=os.path.join(checkpoint_dirpath, 'posenet-{}.pth'.format(step)),
                optimizer_pose=optimizer_pose)
        elif 'voiced' in self.model_name:
            self.model.save_model(
                os.path.join(checkpoint_dirpath, 'voiced-{}.pth'.format(step)),
                step,
                optimizer_depth,
                model_pose_checkpoint_path=os.path.join(checkpoint_dirpath, 'posenet-{}.pth'.format(step)),
                optimizer_pose=optimizer_pose)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(self.model_name))

        # Save continual learning model
        self._get_model_cl().save_model(
            os.path.join(checkpoint_dirpath, 'tokens-{}.pth'.format(step)),
            step,
            optimizer=optimizer_cl)


    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image0=None,
                    image1to0=None,
                    image2to0=None,
                    output_depth0=None,
                    sparse_depth0=None,
                    validity_map0=None,
                    ground_truth0=None,
                    pose0to1=None,
                    pose0to2=None,
                    scalars={},
                    n_image_per_summary=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image0 : torch.Tensor[float32]
                image at time step t
            image1to0 : torch.Tensor[float32]
                image at time step t-1 warped to time step t
            image2to0 : torch.Tensor[float32]
                image at time step t+1 warped to time step t
            output_depth0 : torch.Tensor[float32]
                output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                sparse_depth at time t
            validity_map0 : torch.Tensor[float32]
                validity map of sparse depth at time t
            ground_truth0 : torch.Tensor[float32]
                ground truth depth at time t
            pose0to1 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t-1
            pose0to2 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t+1
            scalars : dict[str, float]
                dictionary of scalars to log
            n_image_per_summary : int
                number of images to display within a summary
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            if image0 is not None:
                image0_summary = image0[0:n_image_per_summary, ...]

                display_summary_image_text += '_image0'
                display_summary_depth_text += '_image0'

                # Normalize for display if necessary
                if torch.max(image0_summary) > 1:
                    image0_summary = image0_summary / 255.0

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image0_summary.cpu(),
                        torch.zeros_like(image0_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            if image0 is not None and image1to0 is not None:
                image1to0_summary = image1to0[0:n_image_per_summary, ...]

                display_summary_image_text += '_image1to0-error'

                # Normalize for display if necessary
                if torch.max(image1to0_summary) > 1:
                    image1to0_summary = image1to0_summary / 255.0

                # Compute reconstruction error w.r.t. image 0
                image1to0_error_summary = torch.mean(
                    torch.abs(image0_summary - image1to0_summary),
                    dim=1,
                    keepdim=True)

                # Add to list of images to log
                image1to0_error_summary = log_utils.colorize(
                    (image1to0_error_summary / 0.10).cpu(),
                    colormap='inferno')

                display_summary_image.append(
                    torch.cat([
                        image1to0_summary.cpu(),
                        image1to0_error_summary],
                        dim=3))

            if image0 is not None and image2to0 is not None:
                image2to0_summary = image2to0[0:n_image_per_summary, ...]

                display_summary_image_text += '_image2to0-error'

                # Normalize for display if necessary
                if torch.max(image2to0_summary) > 1:
                    image2to0_summary = image2to0_summary / 255.0

                # Compute reconstruction error w.r.t. image 0
                image2to0_error_summary = torch.mean(
                    torch.abs(image0_summary - image2to0_summary),
                    dim=1,
                    keepdim=True)

                # Add to list of images to log
                image2to0_error_summary = log_utils.colorize(
                    (image2to0_error_summary / 0.10).cpu(),
                    colormap='inferno')

                display_summary_image.append(
                    torch.cat([
                        image2to0_summary.cpu(),
                        image2to0_error_summary],
                        dim=3))

            if output_depth0 is not None:
                output_depth0_summary = output_depth0[0:n_image_per_summary, ...]

                display_summary_depth_text += '_output0'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth0_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth0_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)

            if output_depth0 is not None and sparse_depth0 is not None and validity_map0 is not None:
                sparse_depth0_summary = sparse_depth0[0:n_image_per_summary, ...]
                validity_map0_summary = validity_map0[0:n_image_per_summary, ...]

                display_summary_depth_text += '_sparse0-error'

                # Compute output error w.r.t. input sparse depth
                sparse_depth0_error_summary = \
                    torch.abs(output_depth0_summary - sparse_depth0_summary)

                sparse_depth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (sparse_depth0_error_summary + 1e-8) / (sparse_depth0_summary + 1e-8),
                    validity_map0_summary)

                # Add to list of images to log
                sparse_depth0_summary = torch.clamp(sparse_depth0_summary, 0.0, self.max_predict_depth)

                sparse_depth0_summary = log_utils.colorize(
                    (sparse_depth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                sparse_depth0_error_summary = log_utils.colorize(
                    (sparse_depth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        sparse_depth0_summary,
                        sparse_depth0_error_summary],
                        dim=3))

                # Log distribution of sparse depth
                summary_writer.add_histogram(tag + '_sparse_depth0_distro', sparse_depth0, global_step=step)

            if output_depth0 is not None and ground_truth0 is not None:

                ground_truth0_summary = ground_truth0[0:n_image_per_summary, ...]
                validity_map0_summary = torch.where(
                    ground_truth0_summary > 0,
                    torch.ones_like(ground_truth0_summary),
                    ground_truth0_summary)

                display_summary_depth_text += '_groundtruth0-error'

                ground_truth0_summary = torch.clamp(ground_truth0_summary, 0.0, self.max_predict_depth)

                # Compute output error w.r.t. ground truth
                ground_truth0_error_summary = \
                    torch.abs(output_depth0_summary - ground_truth0_summary)

                ground_truth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (ground_truth0_error_summary + 1e-8) / (ground_truth0_summary + 1e-8),
                    validity_map0_summary)

                # Add to list of images to log
                ground_truth0_summary = log_utils.colorize(
                    (ground_truth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth0_error_summary = log_utils.colorize(
                    (ground_truth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth0_summary,
                        ground_truth0_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth0_distro', ground_truth0, global_step=step)

            if pose0to1 is not None:
                # Log distribution of pose 1 to 0translation vector
                summary_writer.add_histogram(tag + '_tx0to1_distro', pose0to1[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty0to1_distro', pose0to1[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz0to1_distro', pose0to1[:, 2, 3], global_step=step)

            if pose0to2 is not None:
                # Log distribution of pose 2 to 0 translation vector
                summary_writer.add_histogram(tag + '_tx0to2_distro', pose0to2[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty0to2_distro', pose0to2[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz0to2_distro', pose0to2[:, 2, 3], global_step=step)

        # Log scalars to tensorboard
        for (name, value) in scalars.items():
            summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_image_per_summary),
                global_step=step)

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_image_per_summary),
                global_step=step)
