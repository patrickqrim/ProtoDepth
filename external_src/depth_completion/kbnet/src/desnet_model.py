'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import torch, torchvision
import log_utils, losses, networks, net_utils


EPSILON = 1e-8


class DesNetModel(object):
    '''
    Calibrated Backprojection Network class

    Arg(s):
        input_channels_image : int
            number of channels in the image
        input_channels_depth : int
            number of channels in depth map
        min_pool_sizes_sparse_to_dense_pool : list[int]
            list of min pool kernel sizes for sparse to dense pool
        max_pool_sizes_sparse_to_dense_pool : list[int]
            list of max pool kernel sizes for sparse to dense pool
        n_convolution_sparse_to_dense_pool : int
            number of layers to learn trade off between kernel sizes and near and far structures
        n_filter_sparse_to_dense_pool : int
            number of filters to use in each convolution in sparse to dense pool
        n_filters_encoder_image : list[int]
            number of filters to use in each block of image encoder
        n_filters_encoder_depth : list[int]
            number of filters to use in each block of depth encoder
        resolutions_backprojection : list[int]
            list of resolutions to apply calibrated backprojection
        n_filters_decoder : list[int]
            number of filters to use in each block of depth decoder
        deconv_type : str
            deconvolution types: transpose, up
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        min_predict_depth : float
            minimum predicted depth
        max_predict_depth : float
            maximum predicted depth
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 input_channels_image,
                 input_channels_depth,
                 min_pool_sizes_sparse_to_dense_pool,
                 max_pool_sizes_sparse_to_dense_pool,
                 n_convolution_sparse_to_dense_pool,
                 n_filter_sparse_to_dense_pool,
                 n_filters_encoder_image,
                 n_filters_encoder_depth,
                 resolutions_backprojection,
                 n_filters_decoder,
                 deconv_type='up',
                 weight_initializer='xavier_normal',
                 activation_func='leaky_relu',
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        self.device = device

        # Build sparse to dense pooling
        self.sparse_to_dense_pool = networks.SparseToDensePool(
            input_channels=input_channels_depth,
            min_pool_sizes=min_pool_sizes_sparse_to_dense_pool,
            max_pool_sizes=max_pool_sizes_sparse_to_dense_pool,
            n_convolution=n_convolution_sparse_to_dense_pool,
            n_filter=n_filter_sparse_to_dense_pool,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        # Set up number of input and skip channels
        input_channels_depth = n_filter_sparse_to_dense_pool

        n_filters_encoder = [
            i + z
            for i, z in zip(n_filters_encoder_image, n_filters_encoder_depth)
        ]

        n_skips = n_filters_encoder[:-1]
        n_skips = n_skips[::-1] + [0]

        n_convolutions_encoder_image = [1, 1, 1, 1, 1]
        n_convolutions_encoder_depth = [1, 1, 1, 1, 1]
        n_convolutions_encoder_fused = [1, 1, 1, 1, 1]

        n_filters_encoder_fused = n_filters_encoder_image.copy()

        # Build depth completion network
        self.encoder = networks.KBNetEncoder(
            input_channels_image=input_channels_image,
            input_channels_depth=input_channels_depth,
            n_filters_image=n_filters_encoder_image,
            n_filters_depth=n_filters_encoder_depth,
            n_filters_fused=n_filters_encoder_fused,
            n_convolutions_image=n_convolutions_encoder_image,
            n_convolutions_depth=n_convolutions_encoder_depth,
            n_convolutions_fused=n_convolutions_encoder_fused,
            resolutions_backprojection=resolutions_backprojection,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.decoder = networks.MultiScaleDecoder(
            input_channels=n_filters_encoder[-1],
            output_channels=1,
            n_resolution=1,
            n_filters=n_filters_decoder,
            n_skips=n_skips,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            output_func='linear',
            use_batch_norm=False,
            deconv_type=deconv_type)

        self.scale_decoder = networks.MultiScaleDecoder(
            input_channels=n_filters_encoder[-1],
            output_channels=1,
            n_resolution=1,
            n_filters=n_filters_decoder,
            n_skips=n_skips,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            output_func='linear',
            use_batch_norm=False,
            deconv_type=deconv_type)

        # Move to device
        self.to(self.device)

    def forward(self,
                image,
                sparse_depth,
                validity_map_depth,
                intrinsics):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse depth
            validity_map_depth : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
        Returns:
            torch.Tensor[float32] : N x 1 x H x W output dense depth
        '''

        # Clamp max value of sparse depth
        sparse_depth = torch.where(
            sparse_depth > self.max_predict_depth,
            torch.full_like(sparse_depth, fill_value=self.max_predict_depth),
            sparse_depth)

        # Depth inputs to network:
        # (1) raw sparse depth, (2) filtered validity map
        input_depth = [
            sparse_depth,
            validity_map_depth
        ]

        input_depth = torch.cat(input_depth, dim=1)

        input_depth = self.sparse_to_dense_pool(input_depth)

        # Forward through the network
        shape = input_depth.shape[-2:]
        latent, skips = self.encoder(image, input_depth, intrinsics)

        output = self.decoder(latent, skips, shape)[-1]

        output_depth = torch.sigmoid(output)

        output_scale = self.scale_decoder(latent, skips, shape)[-1]
        # don't know what activation function they used but since scale is non-negative going to use relu here
        output_scale = torch.nn.ReLU()(output_scale) 
        output_scale = torch.nn.functional.adaptive_avg_pool2d(output_scale, (4, 4))
        output_scale = torch.nn.functional.interpolate(output_scale, output_depth.shape[-2:], mode='nearest')
        output_depth = output_depth * output_scale

        return output_depth

    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     sparse_depth0,
                     validity_map_depth0,
                     intrinsics,
                     pose0to1,
                     pose0to2,
                     w_color=0.15,
                     w_structure=0.95,
                     w_sparse_depth=0.60,
                     w_smoothness=0.04,
                     validity_map_image0=None):
        '''
        Computes loss function
        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at time step t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at time step t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W image at time step t+1
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth at time t
            validity_map_depth0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth at time t
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
            validity_map_image0 : torch.Tensor[float32]
                N x 1 x H x W validity map of image at time t
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        shape = image0.shape
        validity_map_image0 = \
            torch.ones_like(sparse_depth0) if validity_map_image0 is None else validity_map_image0

        # Clamp max value of sparse depth
        sparse_depth0 = torch.where(
            sparse_depth0 > self.max_predict_depth,
            torch.full_like(sparse_depth0, fill_value=self.max_predict_depth),
            sparse_depth0)

        # Backproject points to 3D camera coordinates
        points = net_utils.backproject_to_camera(output_depth0, intrinsics, shape)

        # Reproject points onto image 1 and image 2
        target_xy0to1 = net_utils.project_to_pixel(points, pose0to1, intrinsics, shape)
        target_xy0to2 = net_utils.project_to_pixel(points, pose0to2, intrinsics, shape)

        # Reconstruct image0 from image1 and image2 by reprojection
        image1to0 = net_utils.grid_sample(image1, target_xy0to1, shape)
        image2to0 = net_utils.grid_sample(image2, target_xy0to2, shape)

        '''
        Essential loss terms
        '''
        # Color consistency loss function
        loss_color1to0 = losses.color_consistency_loss_func(
            src=image1to0,
            tgt=image0,
            w=validity_map_image0)
        loss_color2to0 = losses.color_consistency_loss_func(
            src=image2to0,
            tgt=image0,
            w=validity_map_image0)
        loss_color = loss_color1to0 + loss_color2to0

        # Structural consistency loss function
        loss_structure1to0 = losses.structural_consistency_loss_func(
            src=image1to0,
            tgt=image0,
            w=validity_map_image0)
        loss_structure2to0 = losses.structural_consistency_loss_func(
            src=image2to0,
            tgt=image0,
            w=validity_map_image0)
        loss_structure = loss_structure1to0 + loss_structure2to0

        # Sparse depth consistency loss function
        loss_sparse_depth = losses.sparse_depth_consistency_loss_func(
            src=output_depth0,
            tgt=sparse_depth0,
            w=validity_map_depth0)

        # Local smoothness loss function
        loss_smoothness = losses.smoothness_loss_func(
            predict=output_depth0,
            image=image0)

        # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}
        loss = w_color * loss_color + \
            w_structure * loss_structure + \
            w_sparse_depth * loss_sparse_depth + \
            w_smoothness * loss_smoothness

        loss_info = {
            'loss_color' : loss_color,
            'loss_structure' : loss_structure,
            'loss_sparse_depth' : loss_sparse_depth,
            'loss_smoothness' : loss_smoothness,
            'loss' : loss,
            'image1to0' : image1to0,
            'image2to0' : image2to0
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list : list of parameters
        '''

        parameters = \
            list(self.sparse_to_dense_pool.parameters()) + \
            list(self.encoder.parameters()) + \
            list(self.decoder.parameters()) + \
            list(self.scale_decoder.parameters())

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.sparse_to_dense_pool.train()
        self.encoder.train()
        self.decoder.train()
        self.scale_decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.sparse_to_dense_pool.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.scale_decoder.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.encoder.to(device)
        self.decoder.to(device)
        self.sparse_to_dense_pool.to(device)
        self.scale_decoder.to(device)

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

        checkpoint = {}
        # Save training state
        checkpoint['train_step'] = step
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save encoder weights
        if isinstance(self.encoder, torch.nn.DataParallel):
            checkpoint['encoder_state_dict'] = self.encoder.module.state_dict()
        else:
            checkpoint['encoder_state_dict'] = self.encoder.state_dict()

        # Save depth decoder weights
        if isinstance(self.decoder, torch.nn.DataParallel):
            checkpoint['decoder_state_dict'] = self.decoder.module.state_dict()
        else:
            checkpoint['decoder_state_dict'] = self.decoder.state_dict()

        # Save sparse-to-dense pooling weights
        if isinstance(self.sparse_to_dense_pool, torch.nn.DataParallel):
            checkpoint['sparse_to_dense_pool_state_dict'] = self.sparse_to_dense_pool.module.state_dict()
        else:
            checkpoint['sparse_to_dense_pool_state_dict'] = self.sparse_to_dense_pool.state_dict()
        
        # save scale decoder
        if isinstance(self.scale_decoder, torch.nn.DataParallel):
            checkpoint['scale_decoder_state_dict'] = self.scale_decoder.module.state_dict()
        else:
            checkpoint['scale_decoder_state_dict'] = self.scale_decoder.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore encoder weights
        if isinstance(self.encoder, torch.nn.DataParallel):
            self.encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        # Restore depth decoder weights
        if isinstance(self.decoder, torch.nn.DataParallel):
            self.decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        # Restore sparse to dense pool
        if isinstance(self.sparse_to_dense_pool, torch.nn.DataParallel):
            self.sparse_to_dense_pool.module.load_state_dict(checkpoint['sparse_to_dense_pool_state_dict'])
        else:
            self.sparse_to_dense_pool.load_state_dict(checkpoint['sparse_to_dense_pool_state_dict'])
        
        # Restore scale decoder
        if isinstance(self.scale_decoder, torch.nn.DataParallel):
            self.scale_decoder.module.load_state_dict(checkpoint['scale_decoder_state_dict'])
        else:
            self.scale_decoder.load_state_dict(checkpoint['scale_decoder_state_dict'])

        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.sparse_to_dense_pool = torch.nn.DataParallel(self.sparse_to_dense_pool)
        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)
        self.scale_decoder = torch.nn.DataParallel(self.scale_decoder)

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
                number of images to display
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
                    (sparse_depth0_error_summary + EPSILON) / (sparse_depth0_summary + EPSILON),
                    validity_map0_summary)

                # Add to list of images to log
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

                n_channel = ground_truth0_summary.shape[1]

                if n_channel == 1:
                    validity_map0_summary = torch.where(
                        ground_truth0 > 0,
                        torch.ones_like(ground_truth0_summary),
                        torch.zeros_like(ground_truth0_summary))
                else:
                    validity_map0_summary = torch.unsqueeze(ground_truth0_summary[:, 1, :, :], dim=1)
                    ground_truth0_summary = torch.unsqueeze(ground_truth0_summary[:, 0, :, :], dim=1)

                display_summary_depth_text += '_groundtruth0-error'

                # Compute output error w.r.t. ground truth
                ground_truth0_error_summary = \
                    torch.abs(output_depth0_summary - ground_truth0_summary)

                ground_truth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (ground_truth0_error_summary + EPSILON) / (ground_truth0_summary + EPSILON),
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
