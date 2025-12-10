import torch, random
import torchvision
import torchvision.transforms.functional as functional
import numpy as np
import copy
from PIL import Image


class Transforms(object):

    def __init__(self,
                 normalized_image_range=None,
                 # Photometric intensity transformations
                 random_brightness=[-1, -1],
                 random_contrast=[-1, -1],
                 random_gamma=[-1, -1],
                 random_hue=[-1, -1],
                 random_saturation=[-1, -1],
                 random_gaussian_blur_kernel_size=[-1, -1],
                 random_gaussian_blur_sigma_range=[-1, -1],
                 random_noise_type='none',
                 random_noise_spread=-1,
                 # Occlusion transformations
                 random_remove_patch_percent_range=[-1, -1],
                 random_remove_patch_size=[1, 1],
                 # Geometric transformations
                 random_crop_to_shape=[-1, -1],
                 random_flip_type=['none'],
                 random_rotate_max=0,
                 random_crop_and_pad=[-1, -1],
                 random_resize_to_shape=[-1, -1],
                 random_resize_and_crop=[-1, -1],
                 random_resize_and_pad=[-1, -1]):
        '''
        Transforms and augmentation class

        Note: brightness, contrast, gamma, hue, saturation augmentations expect
        either type int in [0, 255] or float in [0, 1]

        Arg(s):
            normalized_image_range : list[float]
                intensity range after normalizing images
            random_brightness : list[float]
                brightness adjustment [0, B], from 0 (black image) to B factor increase
            random_contrast : list[float]
                contrast adjustment [0, C], from 0 (gray image) to C factor increase
            random_gamma : list[float]
                gamma adjustment [0, G] from 0 dark to G bright
            random_hue : list[float]
                hue adjustment [-0.5, 0.5] where -0.5 reverses hue to its complement, 0 does nothing
            random_saturation : list[float]
                saturation adjustment [0, S], from 0 (black image) to S factor increase
            random_gaussian_blur_kernel_size : list[int],
                list of kernel sizes to be used for gaussian blur
            random_gaussian_blur_sigma_range : list[float]
                min and max standard deviation (sigma)
            random_noise_type : str
                type of noise to add: gaussian, uniform
            random_noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
            random_remove_patch_percent_range : list[float]
                min and max percentage of points to remove
            random_remove_patch_size : list[int]
                if given [h, w] then constant patch size, if [h1, w1, h2, w2] then min and max kernel size
            random_crop_to_shape : list[int]
                if given [h, w] then output shape after random crop, if [h1, w1, h2, w2] then min and max shape size
            random_flip_type : list[str]
                none, horizontal, vertical
            random_rotate_max : float
                symmetric min and max amount to rotate
            random_crop_and_pad : list[int]
                min and max percentage to crop for random crop and pad
            random_resize_to_shape : list[float]
                min and max percentrage to resize entire batch for random resize
            random_resize_and_crop : list[float]
                min and max percentrage to resize for random resize and will crop back to original shape
            random_resize_and_pad : list[float]
                min and max percentage to resize for random resize and pad back to original shape
        '''

        # Image normalization
        if normalized_image_range is None:
            self.normalized_image_range = None
        elif len(normalized_image_range) > 2:
            mean_stddev_split_idx = len(normalized_image_range) // 2

            self.normalized_image_range = [
                tuple(normalized_image_range[:mean_stddev_split_idx]),
                tuple(normalized_image_range[mean_stddev_split_idx:])
            ]
        else:
            self.normalized_image_range = normalized_image_range

        # Photometric intensity transforms
        self.do_random_brightness = True if -1 not in random_brightness else False
        self.random_brightness = random_brightness

        self.do_random_contrast = True if -1 not in random_contrast else False
        self.random_contrast = random_contrast

        self.do_random_gamma = True if -1 not in random_gamma else False
        self.random_gamma = random_gamma

        self.do_random_hue = True if -1 not in random_hue else False
        self.random_hue = random_hue

        self.do_random_saturation = True if -1 not in random_saturation else False
        self.random_saturation = random_saturation

        self.do_photometric_transforms = \
            self.do_random_brightness or \
            self.do_random_contrast or \
            self.do_random_hue or \
            self.do_random_saturation

        self.do_random_gaussian_blur = \
            True if (-1 not in random_gaussian_blur_kernel_size and -1 not in random_gaussian_blur_sigma_range) else False

        self.random_gaussian_blur_kernel_size = random_gaussian_blur_kernel_size
        self.random_gaussian_blur_sigma_range = random_gaussian_blur_sigma_range

        self.do_random_noise = \
            True if (random_noise_type != 'none' and random_noise_spread > -1) else False

        self.do_image_normalization = self.normalized_image_range is not None

        self.random_noise_type = random_noise_type
        self.random_noise_spread = random_noise_spread

        # Occlusion transforms
        self.do_random_remove_patch = True if -1 not in random_remove_patch_percent_range else False
        self.random_remove_patch_percent_range = random_remove_patch_percent_range

        # Allow random choice of patch size
        if len(random_remove_patch_size) == 4:
            self.random_remove_patch_size_height = list(range(random_remove_patch_size[0], random_remove_patch_size[2] + 2, 2))
            self.random_remove_patch_size_width = list(range(random_remove_patch_size[1], random_remove_patch_size[3] + 2, 2))
        else:
            self.random_remove_patch_size_height = [random_remove_patch_size[0]]
            self.random_remove_patch_size_width = [random_remove_patch_size[1]]

        # Geometric transforms
        self.do_random_crop_to_shape = True if -1 not in random_crop_to_shape else False

        self.do_random_crop_to_shape_exact = False
        self.do_random_crop_to_shape_range = False

        if self.do_random_crop_to_shape:
            if len(random_crop_to_shape) == 2:
                # If performed, will only crop to one shape
                self.do_random_crop_to_shape_exact = True
                self.random_crop_to_shape_height = random_crop_to_shape[0]
                self.random_crop_to_shape_width = random_crop_to_shape[1]
            elif len(random_crop_to_shape) == 4:
                # If performed will crop to any shape between min and max shapes
                self.do_random_crop_to_shape_range = True
                self.random_crop_to_shape_height_min = random_crop_to_shape[0]
                self.random_crop_to_shape_width_min = random_crop_to_shape[1]
                self.random_crop_to_shape_height_max = random_crop_to_shape[2]
                self.random_crop_to_shape_width_max = random_crop_to_shape[3]
            else:
                raise ValueError('Unsupported input for random crop to shape: {}'.format(
                    random_crop_to_shape))

        self.do_random_horizontal_flip = True if 'horizontal' in random_flip_type else False
        self.do_random_vertical_flip = True if 'vertical' in random_flip_type else False

        self.do_random_rotate = True if random_rotate_max > 0 else False
        self.random_rotate_max = random_rotate_max

        self.do_random_crop_and_pad = True if -1 not in random_crop_and_pad else False

        self.random_crop_and_pad_min = random_crop_and_pad[0]
        self.random_crop_and_pad_max = random_crop_and_pad[1]

        if self.do_random_crop_and_pad:
            assert self.random_crop_and_pad_min < self.random_crop_and_pad_max
            assert self.random_crop_and_pad_max <= 1

        self.do_random_resize_to_shape = True if -1 not in random_resize_to_shape else False

        self.random_resize_to_shape_min = random_resize_to_shape[0]
        self.random_resize_to_shape_max = random_resize_to_shape[1]

        if self.do_random_resize_to_shape:
            assert self.random_resize_to_shape_min < self.random_resize_to_shape_max
            assert self.random_resize_to_shape_min > 0

        self.do_random_resize_and_crop = True if -1 not in random_resize_and_crop else False

        self.random_resize_and_crop_min = random_resize_and_crop[0]
        self.random_resize_and_crop_max = random_resize_and_crop[1]

        if self.do_random_resize_and_crop:
            assert self.random_resize_and_crop_min < self.random_resize_and_crop_max
            assert self.random_resize_and_crop_min >= 1.0

        self.do_random_resize_and_pad = True if -1 not in random_resize_and_pad else False

        self.random_resize_and_pad_min = random_resize_and_pad[0]
        self.random_resize_and_pad_max = random_resize_and_pad[1]

        if self.do_random_resize_and_pad:
            assert self.random_resize_and_pad_min < self.random_resize_and_pad_max
            assert self.random_resize_and_pad_min > 0
            assert self.random_resize_and_pad_max <= 1.0

        self.__interpolation_mode_map = {
            'nearest' : Image.NEAREST,
            'bilinear' : Image.BILINEAR,
            'bicubic' : Image.BICUBIC,
            'lanczos' : Image.ANTIALIAS
        }

    def transform(self,
                  images_arr,
                  intrinsics_arr=[],
                  padding_modes=['constant'],
                  interpolation_modes=[Image.NEAREST],
                  random_transform_probability=0.00):
        '''
        Applies transform to images and ground truth

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            intrinsics_arr : list[torch.Tensor]
                list of N x 3 x 3 tensors
            padding_modes : list[str]
                list of padding modes for each tensor in images arr: 'constant', 'edge', 'reflect' or 'symmetric'
            interpolation_modes : list[str]
                list of interpolation modes for each tensor in images arr: 'nearest', 'bilinear', 'bicubic', 'lanczos'
            random_transform_probability : float
                probability to perform transform
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
            list[torch.Tensor[float32]] : list of transformed N x 3 x 3 intrinsics tensors
        '''

        device = images_arr[0].device

        n_dim = images_arr[0].ndim

        transform_performed = {}
        transform_performed['interpolation_modes'] = interpolation_modes

        if n_dim == 4:
            n_batch, n_channel, n_height, n_width = images_arr[0].shape
        elif n_dim == 5:
            n_batch, _, n_channel, n_height, n_width = images_arr[0].shape
        else:
            raise ValueError('Unsupported number of dimensions: {}'.format(n_dim))

        # Roll to do transformation
        do_random_transform = \
            torch.rand(n_batch, device=device) <= random_transform_probability

        '''
        Photometric transforms
        '''
        if self.do_photometric_transforms:
            for idx, images in enumerate(images_arr):
                # In case user pass in image as float type
                if torch.is_floating_point(images):
                    images_arr[idx] = images.to(torch.uint8)

        if self.do_random_brightness:

            do_brightness = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            values = torch.rand(n_batch, device=device)

            brightness_min, brightness_max = self.random_brightness
            factors = (brightness_max - brightness_min) * values + brightness_min

            images_arr = self.adjust_brightness(images_arr, do_brightness, factors)

        if self.do_random_contrast:

            do_contrast = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            contrast_min, contrast_max = self.random_contrast
            factors = (contrast_max - contrast_min) * values + contrast_min

            images_arr = self.adjust_contrast(images_arr, do_contrast, factors)

        if self.do_random_gamma:

            do_gamma = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            gamma_min, gamma_max = self.random_gamma
            gammas = (gamma_max - gamma_min) * values + gamma_min

            images_arr = self.adjust_gamma(images_arr, do_gamma, gammas)

        if self.do_random_hue:

            do_hue = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            hue_min, hue_max = self.random_hue
            factors = (hue_max - hue_min) * values + hue_min

            images_arr = self.adjust_hue(images_arr, do_hue, factors)

        if self.do_random_saturation:

            do_saturation = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            saturation_min, saturation_max = self.random_saturation
            factors = (saturation_max - saturation_min) * values + saturation_min

            images_arr = self.adjust_saturation(images_arr, do_saturation, factors)

        if n_channel == 1:
            images_arr = [
                images[..., 0:1, :, :] for images in images_arr
            ]

        # Convert all images to float
        images_arr = [
            images.float() for images in images_arr
        ]

        if self.do_random_gaussian_blur:

            do_gaussian_blur = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.25)

            values = np.random.rand(n_batch)

            sigma_min, sigma_max = self.random_gaussian_blur_sigma_range

            sigmas = \
                (sigma_max - sigma_min) * values + sigma_min

            kernel_sizes = []

            for _ in range(n_batch):
                kernel_size = random.choice(self.random_gaussian_blur_kernel_size)
                kernel_sizes.append(kernel_size)

            images_arr = self.gaussian_blur(
                images_arr=images_arr,
                do_gaussian_blur=do_gaussian_blur,
                kernel_sizes=kernel_sizes,
                sigmas=sigmas)

        if self.do_random_noise:

            do_add_noise = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.add_noise(
                images_arr,
                do_add_noise=do_add_noise,
                noise_type=self.random_noise_type,
                noise_spread=self.random_noise_spread)

        # Normalize images to a given range
        if self.do_image_normalization:
            images_arr = self.normalize_images(
                images_arr,
                normalized_image_range=self.normalized_image_range)

        '''
        Geometric transforms
        '''
        do_random_crop_to_shape = \
            self.do_random_crop_to_shape and torch.rand(1, device=device) <= 0.50 or \
            self.do_random_crop_to_shape_range

        if do_random_crop_to_shape:

            if self.do_random_crop_to_shape_exact:
                random_crop_to_shape_height = self.random_crop_to_shape_height
                random_crop_to_shape_width = self.random_crop_to_shape_width

            if self.do_random_crop_to_shape_range:
                random_crop_to_shape_height = np.random.randint(
                    low=self.random_crop_to_shape_height_min,
                    high=self.random_crop_to_shape_height_max + 1)

                random_crop_to_shape_width = np.random.randint(
                    low=self.random_crop_to_shape_width_min,
                    high=self.random_crop_to_shape_width_max + 1)

            # Random crop factors
            start_y = torch.randint(
                low=0,
                high=n_height - random_crop_to_shape_height + 1,
                size=(n_batch,),
                device=device)

            start_x = torch.randint(
                low=0,
                high=n_width - random_crop_to_shape_width + 1,
                size=(n_batch,),
                device=device)

            end_y = start_y + random_crop_to_shape_height
            end_x = start_x + random_crop_to_shape_width

            start_yx = [start_y, start_x]
            end_yx = [end_y, end_x]

            images_arr = self.crop(
                images_arr,
                start_yx=start_yx,
                end_yx=end_yx)

            intrinsics_arr = self.adjust_intrinsics(
                intrinsics_arr,
                x_offsets=float(n_width - random_crop_to_shape_width),
                y_offsets=float(n_height - random_crop_to_shape_height))

            # Update shape of tensors after crop
            n_height = random_crop_to_shape_height
            n_width = random_crop_to_shape_width

            transform_performed['random_crop_to_shape'] = [start_yx, end_yx]

        if self.do_random_horizontal_flip:

            do_horizontal_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.horizontal_flip(
                images_arr,
                do_horizontal_flip)

            transform_performed['random_horizontal_flip'] = do_horizontal_flip

        if self.do_random_vertical_flip:

            do_vertical_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.vertical_flip(
                images_arr,
                do_vertical_flip)

            transform_performed['random_vertical_flip'] = do_vertical_flip

        if self.do_random_resize_and_crop:

            do_resize_and_crop = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            # Random resize factors
            scale = torch.rand((n_batch,), device=device)
            scale = (self.random_resize_and_crop_max - self.random_resize_and_crop_min) * scale + \
                self.random_resize_and_crop_min

            r_height = (n_height * scale).int()
            r_width = (n_width * scale).int()

            resize_shape = [r_height, r_width]

            # Adjust intrinsics for resizing factors
            intrinsics_arr = self.adjust_intrinsics(
                intrinsics_arr,
                x_scales=scale,
                y_scales=scale)

            # Get random start (y, x) in resized image
            start_y = []
            start_x = []

            for n in range(n_batch):

                # Get height and width
                height = r_height[n]
                width = r_width[n]

                # Random crop factors
                y = torch.randint(
                    low=0,
                    high=height - n_height + 1,
                    size=(1,),
                    device=device)

                x = torch.randint(
                    low=0,
                    high=width - n_width + 1,
                    size=(1,),
                    device=device)

                start_y.append(y)
                start_x.append(x)

            # Stack them together
            start_y = torch.cat(start_y, dim=0)
            start_x = torch.cat(start_x, dim=0)

            # Get end (y, x) in resized image
            end_y = start_y + n_height
            end_x = start_x + n_width

            start_yx = [start_y, start_x]
            end_yx = [end_y, end_x]

            images_arr = self.resize_and_crop(
                images_arr,
                do_resize_and_crop=do_resize_and_crop,
                resize_shape=resize_shape,
                start_yx=start_yx,
                end_yx=end_yx,
                interpolation_modes=interpolation_modes)

            # Adjust intrinsics for crop offsets
            intrinsics_arr = self.adjust_intrinsics(
                intrinsics_arr,
                x_offsets=(r_width - n_width),
                y_offsets=(r_height - n_height))

            transform_performed['random_resize_and_crop'] = [
                do_resize_and_crop,
                resize_shape,
                start_yx,
                end_yx,
                scale
            ]

        if self.do_random_crop_and_pad:

            do_crop_and_pad = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            # Random crop factors
            max_height = int(self.random_crop_and_pad_max * n_height)
            min_height = int(self.random_crop_and_pad_min * n_height)
            max_width = int(self.random_crop_and_pad_max * n_width)
            min_width = int(self.random_crop_and_pad_min * n_width)

            rand_height = torch.randint(
                low=min_height,
                high=max_height,
                size=(n_batch,),
                device=device)

            rand_width = torch.randint(
                low=min_width,
                high=max_width,
                size=(n_batch,),
                device=device)

            start_y = torch.cat([
                torch.randint(
                    low=0,
                    high=max_height - rh.item(),
                    size=(1,),
                    device=device)
                for rh in rand_height])

            start_x = torch.cat([
                torch.randint(
                    low=0,
                    high=max_width - rw.item(),
                    size=(1,),
                    device=device)
                for rw in rand_width])

            end_y = start_y + rand_height
            end_x = start_x + rand_width

            end_y = torch.minimum(end_y, torch.full_like(end_y, fill_value=n_height))
            end_x = torch.minimum(end_x, torch.full_like(end_x, fill_value=n_width))

            start_yx = [start_y, start_x]
            end_yx = [end_y, end_x]

            # Random padding along all sizes
            d_height = (n_height - (end_y - start_y)).int()
            pad_top = (d_height * torch.rand(n_batch, device=device)).int()
            pad_bottom = d_height - pad_top

            d_width = (n_width - (end_x - start_x)).int()
            pad_left = (d_width * torch.rand(n_batch, device=device)).int()
            pad_right = d_width - pad_left

            padding = [pad_top, pad_bottom, pad_left, pad_right]

            images_arr = self.crop_and_pad(
                images_arr,
                do_crop_and_pad=do_crop_and_pad,
                start_yx=start_yx,
                end_yx=end_yx,
                padding=padding,
                padding_modes=padding_modes)

            transform_performed['random_crop_and_pad'] = [
                do_crop_and_pad,
                start_yx,
                end_yx,
                padding
            ]

        if self.do_random_resize_and_pad:

            do_resize_and_pad = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            # Random resize factors
            scale = torch.rand((n_batch,), device=device)
            scale = (self.random_resize_and_pad_max - self.random_resize_and_pad_min) * scale + \
                self.random_resize_and_pad_min

            r_height = (n_height * scale).int()
            r_width = (n_width * scale).int()

            shape = [r_height, r_width]

            # Random padding along all sizes
            d_height = (n_height - r_height).int()
            pad_top = (d_height * torch.rand(n_batch, device=device)).int()
            pad_bottom = d_height - pad_top

            d_width = (n_width - r_width).int()
            pad_left = (d_width * torch.rand(n_batch, device=device)).int()
            pad_right = d_width - pad_left

            pad_top = torch.maximum(pad_top, torch.full_like(pad_top, fill_value=0))
            pad_bottom = torch.maximum(pad_bottom, torch.full_like(pad_bottom, fill_value=0))
            pad_left = torch.maximum(pad_left, torch.full_like(pad_left, fill_value=0))
            pad_right = torch.maximum(pad_right, torch.full_like(pad_right, fill_value=0))

            padding = [pad_top, pad_bottom, pad_left, pad_right]

            images_arr = self.resize_and_pad(
                images_arr,
                do_resize_and_pad=do_resize_and_pad,
                resize_shape=shape,
                max_shape=(n_height, n_width),
                padding=padding,
                padding_modes=padding_modes,
                interpolation_modes=interpolation_modes)

            # Adjust intrinsics for crop offsets
            intrinsics_arr = self.adjust_intrinsics(
                intrinsics_arr,
                x_scales=scale,
                y_scales=scale,
                x_offsets=-pad_left,
                y_offsets=-pad_top)

            transform_performed['random_resize_and_pad'] = [
                do_resize_and_pad,
                shape,
                padding,
                scale
            ]

        if self.do_random_rotate:

            do_rotate = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.05)

            values = np.random.rand(n_batch)

            rotate_min = -self.random_rotate_max
            rotate_max = self.random_rotate_max

            angles = (rotate_max - rotate_min) * values + rotate_min

            original_shapes = [x.shape[-2:] for x in images_arr]

            images_arr = self.rotate(
                images_arr=images_arr,
                do_rotate=do_rotate,
                angles=angles,
                padding_modes=padding_modes,
                interpolation_modes=interpolation_modes)

            transform_performed['random_rotate'] = [
                do_rotate,
                angles,
                original_shapes]

        if self.do_random_resize_to_shape and torch.rand([], device=device) <= 0.50:

            # Random resize factors
            range_resize_to_shape = self.random_resize_to_shape_max - self.random_resize_to_shape_min

            resize_factor = \
                float(self.random_resize_to_shape_min + range_resize_to_shape * np.random.rand())

            shape = [int(n_height * resize_factor), int(n_width * resize_factor)]

            images_arr = self.resize_to_shape(
                images_arr,
                resize_shape=shape,
                interpolation_modes=interpolation_modes)

            # Adjust intrinsics for resizing factors
            intrinsics_arr = self.adjust_intrinsics(
                intrinsics_arr,
                x_scales=resize_factor,
                y_scales=resize_factor)

            transform_performed['random_resize_to_shape'] = [(n_height, n_width), resize_factor]

            n_height, n_width = shape

        '''
        Occlusion transforms
        '''
        if self.do_random_remove_patch:

            do_remove_points = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            remove_percent_min, remove_percent_max = self.random_remove_patch_percent_range

            densities = \
                (remove_percent_max - remove_percent_min) * values + remove_percent_min

            patch_sizes = []

            for _ in range(n_batch):
                patch_size = [
                    random.choice(self.random_remove_patch_size_height),
                    random.choice(self.random_remove_patch_size_width)
                ]
                patch_sizes.append(patch_size)

            images_arr = self.remove_random_patches(
                images_arr=images_arr,
                do_remove=do_remove_points,
                densities=densities,
                patch_sizes=patch_sizes)

        outputs = []

        if len(images_arr) > 0:
            outputs.append(images_arr)

        if len(intrinsics_arr) > 0:
            outputs.append(intrinsics_arr)

        outputs.append(transform_performed)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def reverse_transform(self,
                          images_arr,
                          transform_performed,
                          padding_modes=['edge'],
                          return_all_outputs=True,
                          do_scale_reverse=False):
        '''
        Reverses the transformations applied to images in the transform method.
        '''
        # order of the transformations matters here. Be sure to transform it in the reverse order

        # Make sure all transforms are reversed
        transform_performed = copy.deepcopy(transform_performed)

        if 'random_resize_to_shape' in transform_performed:
            shape, _ = transform_performed.pop('random_resize_to_shape')

            images_arr = self.reverse_resize_to_shape(
                images_arr=images_arr,
                resize_shape=shape,
                interpolation_modes=transform_performed['interpolation_modes'])

        if 'random_rotate' in transform_performed:
            do_rotate, angles, original_shapes = transform_performed.pop('random_rotate')

            images_arr = self.reverse_rotate(
                images_arr,
                do_rotate,
                original_shapes,
                angles=angles,
                interpolation_modes=transform_performed['interpolation_modes'],
                return_all_outputs=True)

        validity_maps_arr = [
            torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) for x in images_arr
        ]

        # reversing resize and pad
        if 'random_resize_and_pad' in transform_performed:
            do_resize_and_pad, shape, padding, scale = transform_performed.pop('random_resize_and_pad')

            images_arr = self.reverse_resize_and_pad(
                images_arr,
                do_resize_and_pad,
                resize_shape=shape,
                padding=padding,
                scale=scale,
                do_scale_reverse=do_scale_reverse,
                padding_modes=padding_modes,
                interpolation_modes=transform_performed['interpolation_modes'])

        if 'random_crop_and_pad' in transform_performed:
            do_crop_and_pad, start_yx, end_yx, padding = transform_performed.pop('random_crop_and_pad')

            images_arr, new_validity_maps_arr = self.reverse_crop_and_pad(
                images_arr,
                do_crop_and_pad,
                start_yx,
                end_yx,
                padding,
                padding_modes=padding_modes)

            for i, validity_maps in enumerate(validity_maps_arr):
                validity_maps_arr[i] = validity_maps * new_validity_maps_arr[i]

        if 'random_resize_and_crop' in transform_performed:
            do_resize_and_crop, resize_shape, start_yx, end_yx, scale = transform_performed.pop('random_resize_and_crop')

            images_arr, new_validity_maps_arr = self.reverse_resize_and_crop(
                images_arr,
                do_resize_and_crop,
                resize_shape,
                start_yx,
                end_yx,
                scale,
                do_scale_reverse=do_scale_reverse,
                padding_modes=padding_modes,
                interpolation_modes=transform_performed['interpolation_modes'])

            for i, validity_maps in enumerate(validity_maps_arr):
                validity_maps_arr[i] = validity_maps * new_validity_maps_arr[i]

        if 'random_vertical_flip' in transform_performed:
            do_vertical_flip = transform_performed.pop('random_vertical_flip')

            images_arr = self.reverse_vertical_flip(
                images_arr,
                do_vertical_flip)

        if 'random_horizontal_flip' in transform_performed:
            do_horizontal_flip = transform_performed.pop('random_horizontal_flip')

            images_arr = self.reverse_horizontal_flip(
                images_arr,
                do_horizontal_flip)

        # Make sure all transforms are popped and reversed
        if 'interpolation_modes' in transform_performed:
            transform_performed.pop('interpolation_modes')

        assert len(transform_performed) == 0

        return images_arr, validity_maps_arr

    '''
    Photometric transforms
    '''
    def normalize_images(self, images_arr, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            images_arr[torch.Tensor[float32]] : list of normalized N x C x H x W tensors
        '''

        do_normalization_standard = any([
            isinstance(value, tuple) or isinstance(value, list)
            for value in normalized_image_range])

        if normalized_image_range == [0, 1]:
            images_arr = [
                images / 255.0 for images in images_arr
            ]

        elif do_normalization_standard:
            # Perform standard normalization
            mean, std = normalized_image_range[0], normalized_image_range[1]
            images_arr = [
                torchvision.transforms.functional.normalize(
                    images / 255.0,
                    mean,
                    std)
                for images in images_arr
            ]

        elif normalized_image_range == [-1, 1]:
            images_arr = [
                2.0 * (images / 255.0) - 1.0 for images in images_arr
            ]
        elif normalized_image_range == [0, 255]:
            pass
        else:
            raise ValueError('Unsupported normalization range: {}'.format(
                normalized_image_range))

        return images_arr

    def adjust_brightness(self, images_arr, do_brightness, factors):
        '''
        Adjust brightness on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_brightness : bool
                N booleans to determine if brightness is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_brightness[b]:
                    images[b, ...] = functional.adjust_brightness(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_contrast(self, images_arr, do_contrast, factors):
        '''
        Adjust contrast on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_contrast : bool
                N booleans to determine if contrast is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_contrast[b]:
                    images[b, ...] = functional.adjust_contrast(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_gamma(self, images_arr, do_gamma, gammas):
        '''
        Adjust gamma on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_gamma : bool
                N booleans to determine if gamma is adjusted on each sample
            gammas : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_gamma[b]:
                    images[b, ...] = functional.adjust_gamma(image, gammas[b], gain=1)

            images_arr[i] = images

        return images_arr

    def adjust_hue(self, images_arr, do_hue, factors):
        '''
        Adjust hue on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_hue : bool
                N booleans to determine if hue is adjusted on each sample
            gammas : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_hue[b]:
                    images[b, ...] = functional.adjust_hue(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_saturation(self, images_arr, do_saturation, factors):
        '''
        Adjust saturation on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_saturation : bool
                N booleans to determine if saturation is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_saturation[b]:
                    images[b, ...] = functional.adjust_saturation(image, factors[b])

            images_arr[i] = images

        return images_arr

    def gaussian_blur(self, images_arr, do_gaussian_blur, kernel_sizes, sigmas):
        '''
        Remove random nonzero for each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_gaussian_blur : bool
                N booleans to determine if random gaussian blur is performed on each sample
            kernel_sizes : list[int]
                list of kernel sizes to use for gaussian blur
            sigmas : list[float]
                N floats to determine the standard deviation (sigma) of the gaussian
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_gaussian_blur[b]:

                    kernel_size = kernel_sizes[b]
                    sigma = sigmas[b]

                    images[b, ...] = torchvision.transforms.functional.gaussian_blur(
                        img=image,
                        kernel_size=kernel_size,
                        sigma=sigma)

            images_arr[i] = images

        return images_arr

    def add_noise(self, images_arr, do_add_noise, noise_type, noise_spread):
        '''
        Add noise to images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_add_noise : bool
                N booleans to determine if noise will be added
            noise_type : str
                gaussian, uniform
            noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
        '''

        for i, images in enumerate(images_arr):
            device = images.device

            for b, image in enumerate(images):
                if do_add_noise[b]:

                    shape = image.shape

                    if noise_type == 'gaussian':
                        image = image + noise_spread * torch.randn(*shape, device=device)
                    elif noise_type == 'uniform':
                        image = image + noise_spread * (torch.rand(*shape, device=device) - 0.5)
                    else:
                        raise ValueError('Unsupported noise type: {}'.format(noise_type))

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    '''
    Occlusion transforms
    '''
    def remove_random_patches(self, images_arr, do_remove, densities, patch_sizes):
        '''
        Remove random nonzero for each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_remove : bool
                N booleans to determine if random remove is performed on each sample
            densities : list[float]
                N floats to determine how much to remove from each sample
            patch_sizes : list[int]
                list of patch (kernel) sizes to remove from each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_remove[b]:

                    patch_size = patch_sizes[b]

                    mask = torch.sum(torch.abs(image), dim=0, keepdim=True)

                    mask = torch.where(
                        mask > 0,
                        torch.ones_like(mask),
                        torch.zeros_like(mask))

                    nonzero_indices = self.random_nonzero(mask, density=densities[b])
                    mask[nonzero_indices] = float('inf')

                    mask = torch.nn.functional.max_pool2d(
                        input=mask,
                        kernel_size=patch_size,
                        stride=1,
                        padding=[int(k // 2) for k in patch_size])

                    mask[mask == float('inf')] = 0.0

                    images[b, ...] = mask * image

            images_arr[i] = images

        return images_arr

    def random_nonzero(self, T, density=0.10):
        '''
        Randomly selects nonzero elements

        Arg(s):
            T : torch.Tensor[float32]
                N x C x H x W tensor
            density : float
                percentage of nonzero elements to select
        Returns:
            list[tuple[torch.Tensor[float32]]] : list of tuples of indices
        '''

        # Find all nonzero indices
        nonzero_indices = (T > 0).nonzero(as_tuple=True)

        # Randomly choose a subset of the indices
        random_subset = torch.randperm(nonzero_indices[0].shape[0], device=T.device)
        random_subset = random_subset[0:int(density * random_subset.shape[0])]

        random_nonzero_indices = [
            indices[random_subset] for indices in nonzero_indices
        ]

        return random_nonzero_indices

    '''
    Geometric transforms
    '''
    def crop(self, images_arr, start_yx, end_yx):
        '''
        Performs cropping on images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            start_yx : list[int, int]
                top left corner y, x coordinate
            end_yx : list
                bottom right corner y, x coordinate
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            images_cropped = []

            for b, image in enumerate(images):

                start_y = start_yx[0][b]
                start_x = start_yx[1][b]
                end_y = end_yx[0][b]
                end_x = end_yx[1][b]

                # Crop image
                image = image[..., start_y:end_y, start_x:end_x]

                images_cropped.append(image)

            images_arr[i] = torch.stack(images_cropped, dim=0)

        return images_arr

    def horizontal_flip(self, images_arr, do_horizontal_flip, in_place=False):
        '''
        Perform horizontal flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_horizontal_flip : bool
                N booleans to determine if horizontal flip is performed on each sample
            in_place : bool
                if set, then will perform horizontal flip in place
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            if in_place:
                output_images = images
            else:
                output_images = torch.zeros_like(images)

            for b, image in enumerate(images):
                if do_horizontal_flip[b]:
                    output_images[b] = torch.flip(image, dims=[-1])
                else:
                    output_images[b] = image

            images_arr[i] = output_images

        return images_arr

    def vertical_flip(self, images_arr, do_vertical_flip, in_place=False):
        '''
        Perform vertical flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_vertical_flip : bool
                N booleans to determine if vertical flip is performed on each sample
            in_place : bool
                if set, then will perform horizontal flip in place
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            if in_place:
                output_images = images
            else:
                output_images = torch.zeros_like(images)

            for b, image in enumerate(images):
                if do_vertical_flip[b]:
                    output_images[b] = torch.flip(image, dims=[-2])
                else:
                    output_images[b] = image

            images_arr[i] = output_images

        return images_arr

    def rotate(self,
               images_arr,
               do_rotate,
               angles,
               padding_modes=['constant'],
               interpolation_modes=[Image.NEAREST]):
        '''
        Rotates each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_rotate : bool
                N booleans to determine if rotation is performed on each sample
            angles : float
                N floats to determine how much to rotate each sample
            padding_modes : list[str]
                list of modes for padding: 'constant', 'edge', 'reflect' or 'symmetric'
            interpolation_modes : list[int]
                list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        n_images_arr = len(images_arr)

        if len(padding_modes) < n_images_arr:
            padding_modes = \
                padding_modes + [padding_modes[-1]] * (n_images_arr - len(padding_modes))

        if len(interpolation_modes) < n_images_arr:
            interpolation_modes = \
                interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))

        for i, (images, padding_mode, interpolation_mode) in enumerate(zip(images_arr, padding_modes, interpolation_modes)):

            rotated_images = [None] * images.shape[0]
            max_image_shape = [-1, -1]

            for b, image in enumerate(images):
                if do_rotate[b]:
                    rotated_images[b] = functional.rotate(
                        image,
                        angle=angles[b],
                        resample=interpolation_mode,
                        expand=True)

                else:
                    rotated_images[b] = image

                if rotated_images[b].shape[1] > max_image_shape[0]:
                    max_image_shape[0] = rotated_images[b].shape[1]
                if rotated_images[b].shape[2] > max_image_shape[1]:
                    max_image_shape[1] = rotated_images[b].shape[2]

            if max_image_shape[0] > images.shape[2] or max_image_shape[1] > images.shape[3]:

                for b, rotated_image in enumerate(rotated_images):
                    rotated_images[b] = self.pad_to_shape(
                        [rotated_image],
                        shape=max_image_shape,
                        padding_mode=padding_mode,
                        padding_value=0)[0]

            images = torch.stack(rotated_images, dim=0)
            images_arr[i] = images

        return images_arr

    def crop_and_pad(self,
                     images_arr,
                     do_crop_and_pad,
                     start_yx,
                     end_yx,
                     padding,
                     padding_modes=['constant'],
                     padding_value=0):
        '''
        Crop and pad images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_crop_and_pad : bool
                N booleans to determine if image will be cropped and padded
            start_yx : list[int, int]
                top left corner y, x coordinate
            end_yx : list
                bottom right corner y, x coordinate
            padding : list[int, int, int, int]
                list of padding for left, top, right, bottom sides
            padding_modes : list[str]
                list of modes for padding: 'constant', 'edge', 'reflect' or 'symmetric'
            padding_value : float
                value to pad with
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        n_images_arr = len(images_arr)

        if len(padding_modes) < n_images_arr:
            padding_modes = padding_modes + [padding_modes[-1]] * (n_images_arr - len(padding_modes))

        for i, (images, padding_mode) in enumerate(zip(images_arr, padding_modes)):

            for b, image in enumerate(images):
                if do_crop_and_pad[b]:

                    start_y = start_yx[0][b]
                    start_x = start_yx[1][b]
                    end_y = end_yx[0][b]
                    end_x = end_yx[1][b]
                    pad_top = padding[0][b]
                    pad_bottom = padding[1][b]
                    pad_left = padding[2][b]
                    pad_right = padding[3][b]

                    # Crop image
                    image = image[..., start_y:end_y, start_x:end_x]

                    # Pad image
                    image = functional.pad(
                        image,
                        (pad_left, pad_top, pad_right, pad_bottom),
                        padding_mode=padding_mode,
                        fill=padding_value)

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    def resize_to_shape(self,
                        images_arr,
                        resize_shape,
                        interpolation_modes=[Image.NEAREST]):
        '''
        Resize all images to shape

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            resize_shape : list[int, int]
                height and width to resize
            interpolation_modes : list[int]
                list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        n_images_arr = len(images_arr)

        if len(interpolation_modes) < n_images_arr:
            interpolation_modes = \
                interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))

        r_height, r_width = resize_shape

        for i, (images, interpolation_mode) in enumerate(zip(images_arr, interpolation_modes)):

            # Resize image
            images = functional.resize(
                images,
                size=(r_height, r_width),
                interpolation=interpolation_mode)

            images_arr[i] = images

        return images_arr

    def resize_and_pad(self,
                       images_arr,
                       do_resize_and_pad,
                       resize_shape,
                       max_shape,
                       padding,
                       padding_modes=['constant'],
                       padding_value=0,
                       interpolation_modes=[Image.NEAREST]):
        '''
        Resize and pad images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_resize_and_pad : bool
                N booleans to determine if image will be resized and padded
            resize_shape : list[int, int]
                height and width to resize
            max_shape : tuple[int]
                max height and width, if exceed center crop
            padding : list[int, int, int, int]
                list of padding for left, top, right, bottom sides
            padding_modes : list[str]
                list of modes for padding: 'constant', 'edge', 'reflect' or 'symmetric'
            padding_value : float
                value to pad with
            interpolation_modes : list[int]
                list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        n_images_arr = len(images_arr)

        if len(padding_modes) < n_images_arr:
            padding_modes = \
                padding_modes + [padding_modes[-1]] * (n_images_arr - len(padding_modes))

        if len(interpolation_modes) < n_images_arr:
            interpolation_modes = \
                interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))

        for i, (images, interpolation_mode, padding_mode) in enumerate(zip(images_arr, interpolation_modes, padding_modes)):

            for b, image in enumerate(images):
                if do_resize_and_pad[b]:

                    r_height = resize_shape[0].cpu().numpy()[b]
                    r_width = resize_shape[1].cpu().numpy()[b]
                    pad_top = padding[0][b]
                    pad_bottom = padding[1][b]
                    pad_left = padding[2][b]
                    pad_right = padding[3][b]

                    # Resize image
                    image = functional.resize(
                        image,
                        size=(r_height, r_width),
                        interpolation=interpolation_mode)

                    # Pad image
                    image = functional.pad(
                        image,
                        (pad_left, pad_top, pad_right, pad_bottom),
                        padding_mode=padding_mode,
                        fill=padding_value)

                    height, width = image.shape[-2:]
                    max_height, max_width = max_shape

                    # If resized image is larger, then do center crop
                    if max_height < height or max_width < width:
                        start_y = height - max_height
                        start_x = width - max_width
                        end_y = start_y + max_height
                        end_x = start_x + max_width
                        image = image[..., start_y:end_y, start_x:end_x]

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    def resize_and_crop(self,
                        images_arr,
                        do_resize_and_crop,
                        resize_shape,
                        start_yx,
                        end_yx,
                        interpolation_modes=[Image.NEAREST]):
        '''
        Resize and crop to shape

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_resize_and_crop : bool
                N booleans to determine if image will be resized and crop based on input (y, x)
            resize_shape : list[int, int]
                height and width to resize
            start_yx : list[int, int]
                top left corner y, x coordinate
            end_yx : list
                bottom right corner y, x coordinate
            interpolation_modes : list[int]
                list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        n_images_arr = len(images_arr)

        if len(interpolation_modes) < n_images_arr:
            interpolation_modes = \
                interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))

        for i, (images, interpolation_mode) in enumerate(zip(images_arr, interpolation_modes)):

            images_cropped = []

            for b, image in enumerate(images):
                if do_resize_and_crop[b]:

                    r_height = resize_shape[0].cpu().numpy()[b]
                    r_width = resize_shape[1].cpu().numpy()[b]

                    # Resize image
                    image = functional.resize(
                        image,
                        size=(r_height, r_width),
                        interpolation=interpolation_mode)

                    start_y = start_yx[0][b]
                    start_x = start_yx[1][b]
                    end_y = end_yx[0][b]
                    end_x = end_yx[1][b]

                    # Crop image
                    image = image[..., start_y:end_y, start_x:end_x]

                    images_cropped.append(image)
                else:
                    images_cropped.append(image)

            images_arr[i] = torch.stack(images_cropped, dim=0)

        return images_arr

    def pad_to_shape(self, images_arr, shape, padding_mode='constant', padding_value=0):
        '''
        Pads images to shape

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            shape : list[int]
                output shape after padding
            padding_mode : str
                list of modes for padding: 'constant', 'edge', 'reflect' or 'symmetric'
            padding_value : float
                value to pad with
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        p_height, p_width = shape

        for i, images in enumerate(images_arr):

            n_height, n_width = images.shape[-2:]
            d_height = max(p_height - n_height, 0)
            d_width = max(p_width - n_width, 0)

            if d_height > 0 or d_width > 0:

                pad_top = d_height // 2
                pad_bottom = d_height - pad_top
                pad_left = d_width // 2
                pad_right = d_width - pad_left

                images = functional.pad(
                    images,
                    (pad_left, pad_top, pad_right, pad_bottom),
                    padding_mode=padding_mode,
                    fill=padding_value)

            images_arr[i] = images

        return images_arr

    '''
    Inverse geometric transforms
    '''
    def reverse_horizontal_flip(self, images_arr, do_horizontal_flip):
        '''
        Reverse horizontal flip for images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_horizontal : list[bool]
                N booleans to determine if image was flipped
        Returns:
            list[torch.Tensor] : list of N x C x H x W tensors
        '''

        return self.horizontal_flip(images_arr, do_horizontal_flip, in_place=False)

    def reverse_vertical_flip(self, images_arr, do_vertical_flip):
        '''
        Reverse horizontal flip for images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_horizontal : list[bool]
                N booleans to determine if image was flipped
        Returns:
            list[torch.Tensor] : list of N x C x H x W tensors
        '''

        return self.vertical_flip(images_arr, do_vertical_flip, in_place=False)

    def reverse_rotate(self,
                       images_arr,
                       do_rotate,
                       original_shapes,
                       angles,
                       interpolation_modes=[Image.NEAREST],
                       return_all_outputs=True):
        '''
        Reverse random rotate for images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_rotate : list[bool]
                N booleans to determine if image will be rotated
            angles : list[float]
                N floats to determine how much to rotate each sample
        Returns:
            list[torch.Tensor] : list of N x C x H x W tensors
        '''

        n_images_arr = len(images_arr)

        if len(interpolation_modes) < n_images_arr:
            interpolation_modes = \
                interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))
        if len(original_shapes) < n_images_arr:
            original_shapes = original_shapes * n_images_arr

        for i, (images, interpolation_mode, original_shape) in enumerate(zip(images_arr, interpolation_modes, original_shapes)):

            reverse_rotated_images = [None] * images.shape[0]

            for b, image in enumerate(images):
                # print(image.shape,do_rotate[b], angles[b], original_shape)
                if do_rotate[b]:
                    reverse_rotated_images[b] = functional.rotate(
                        image,
                        angle=-1 * angles[b],
                        resample=interpolation_mode,
                        expand=True)
                else:
                    reverse_rotated_images[b] = image

                # doing a center crop back to the original shape for the reverse rotated images
                start_y = (reverse_rotated_images[b].shape[1] - original_shape[0]) // 2
                start_x = (reverse_rotated_images[b].shape[2] - original_shape[1]) // 2
                end_y = start_y + original_shape[0]
                end_x = start_x + original_shape[1]

                reverse_rotated_images[b] = reverse_rotated_images[b][:, start_y:end_y, start_x:end_x]

            images = torch.stack(reverse_rotated_images, dim=0)

            images_arr[i] = images

        return images_arr

    def reverse_crop_and_pad(self,
                images_arr,
                do_crop_and_pad,
                start_yx,
                end_yx,
                padding,
                padding_modes=['edge'],
                padding_value=0):
        '''
        Reversing crop and pad images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_crop_and_pad : bool
                N booleans to determine if image will be cropped and padded
            start_yx: list[int, int]
                start y, start x
            end_yx: list[int, int]
                end y, end x
            padding : list[int, int, int, int]
                list of padding for left, top, right, bottom sides
            padding_modes : list[str]
                list of modes for padding: 'constant', 'edge', 'reflect' or 'symmetric'
            padding_value : float
                value to pad with
            interpolation_modes : list[int]
                list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
            list[torch.Tensor] : list of N x 1 x H x W validity maps
        '''

        n_images_arr = len(images_arr)

        if len(padding_modes) < n_images_arr:
            padding_modes = \
                padding_modes + [padding_modes[-1]] * (n_images_arr - len(padding_modes))

        validity_maps_arr = [None] * len(images_arr)

        for i, (images, padding_mode) in enumerate(zip(images_arr, padding_modes)):

            output_images = torch.zeros_like(images)
            validity_maps = torch.ones(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device)

            for b, image in enumerate(images):
                if do_crop_and_pad[b]:

                    pad_top = padding[0][b]
                    pad_bottom = padding[1][b]
                    pad_left = padding[2][b]
                    pad_right = padding[3][b]
                    height, width = image.shape[-2:]

                    # padding performed
                    translated_start_x = pad_left
                    translated_start_y = pad_top
                    translated_end_x = width - pad_right
                    translated_end_y = height - pad_bottom

                    start_y, start_x = start_yx[0][b], start_yx[1][b]
                    endy_y, end_x = end_yx[0][b], end_yx[1][b]

                    right_translation = translated_start_x - start_x
                    bottom_translation = translated_start_y - start_y

                    new_pad_top, new_pad_bottom, new_pad_left, new_pad_right = 0, 0, 0, 0
                    # NOTE: This code tries to remove as little predicted regions as possible
                    # --- Instead of extracting the cropped image and adding back the original
                    #     padding, we only remove regions arising from translation. I.e.
                    #
                    #    1122           0000               qwer     wer0
                    #    3344 --> 34 -> 0034 --(predict)-> asdf --> sdf0    instead of
                    #    5566           0000               zxcv     xcv0
                    #
                    #    1122           0000               qwer            0000
                    #    3344 --> 34 -> 0034 --(predict)-> asdf --> df --> 0df0
                    #    5566           0000               zxcv            0000

                    if right_translation > 0:
                        # Image translated right, need to move it back to the left
                        image = image[:, :, right_translation:]
                        new_pad_right = right_translation

                    elif right_translation < 0:
                        # Image translated left, need to move it back to the right
                        image = image[:, :, :width + right_translation]
                        new_pad_left = -right_translation

                    if bottom_translation > 0:
                        # Image translated down, need to move it back up
                        image = image[ :, bottom_translation:, :]
                        new_pad_bottom = bottom_translation

                    elif bottom_translation < 0:
                        # Image translated up, need to move it back down
                        image = image[:, :height + bottom_translation, :]
                        new_pad_top = -bottom_translation

                    validity_map = torch.ones(1, image.shape[1], image.shape[2], device=image.device)

                    image = functional.pad(
                        image,
                        (new_pad_left, new_pad_top, new_pad_right, new_pad_bottom),
                        fill=padding_value,
                        padding_mode=padding_mode)

                    validity_map = functional.pad(
                        validity_map,
                        (new_pad_left, new_pad_top, new_pad_right, new_pad_bottom),
                        fill=0,
                        padding_mode='constant')

                    validity_maps[b, ...] = validity_map
                output_images[b, ...] = image

            images_arr[i] = output_images
            validity_maps_arr[i] = validity_maps

        return images_arr, validity_maps_arr

    def reverse_resize_to_shape(self,
                                images_arr,
                                resize_shape,
                                interpolation_modes=[Image.NEAREST]):
        '''
        Reversing resize to shape

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            resize_shape : list[int]
                the shape that the image will be resized back to
            interpolation_modes : list[int]
                list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        n_images_arr = len(images_arr)

        if len(interpolation_modes) < n_images_arr:
            interpolation_modes = \
                interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))

        n_height, n_width = resize_shape

        output_images_arr = []

        for images, interpolation_mode in zip(images_arr, interpolation_modes):

            output_images = []

            for image in images:

                output_image = functional.resize(
                    image,
                    size=(n_height, n_width),
                    interpolation=interpolation_mode)

                output_images.append(output_image)

            output_images = torch.stack(output_images, dim=0)
            output_images_arr.append(output_images)

        return output_images_arr

    def reverse_resize_and_crop(self,
                                images_arr,
                                do_resize_and_crop,
                                resize_shape,
                                start_yx,
                                end_yx,
                                scale,
                                do_scale_reverse=True,
                                padding_modes=['edge'],
                                padding_value=0,
                                interpolation_modes=[Image.NEAREST]):
        '''
        Reversing resize and pad images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_resize_and_pad : bool
                N booleans to determine if image was resized and padded
            resize_shape : list[int]
                the shape that the image will be resized back to
            start_yx: list[int, int]
                start y, start x
            end_yx: list[int, int]
                end y, end x
            scale: list[float]
                scale used to re-scale the values of images
            do_scale_reverse : bool
                if set, then re-scale the values of images by scale
            padding : list[int, int, int, int]
                list of padding for left, top, right, bottom sides
            padding_modes : list[str]
                list of modes for padding: 'constant', 'edge', 'reflect' or 'symmetric'
            padding_value : float
                value to pad with
            interpolation_modes : list[int]
                list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
            list[torch.Tensor] : list of N x 1 x H x W validity maps
        '''

        n_images_arr = len(images_arr)

        if len(padding_modes) < n_images_arr:
            padding_modes = \
                padding_modes + [padding_modes[-1]] * (n_images_arr - len(padding_modes))

        if len(interpolation_modes) < n_images_arr:
            interpolation_modes = \
                interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))

        validity_maps_arr = [None] * len(images_arr)

        for i, (images, padding_mode, interpolation_mode) in enumerate(zip(images_arr, padding_modes, interpolation_modes)):

            output_images = torch.zeros_like(images)
            validity_maps = torch.ones(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device)

            for b, image in enumerate(images):
                output_image = image
                if do_resize_and_crop[b]:
                    _, n_height, n_width = image.shape

                    pad_left = start_yx[1][b]
                    pad_right = resize_shape[1][b] - end_yx[1][b]
                    pad_top = start_yx[0][b]
                    pad_bottom = resize_shape[0][b] - end_yx[0][b]

                    if do_scale_reverse:
                        output_image = output_image * scale[b]

                    validity_map = torch.ones(1, image.shape[1], image.shape[2], device=image.device)

                    output_image = functional.pad(
                        output_image,
                        (pad_left, pad_top, pad_right, pad_bottom),
                        padding_mode=padding_mode,
                        fill=padding_value)

                    validity_map = functional.pad(
                        validity_map,
                        (pad_left, pad_top, pad_right, pad_bottom),
                        padding_mode='constant',
                        fill=0)

                    output_image = functional.resize(
                        output_image,
                        size=(n_height, n_width),
                        interpolation=interpolation_mode)

                    validity_map = functional.resize(
                        validity_map,
                        size=(n_height, n_width),
                        interpolation=Image.NEAREST)

                    validity_maps[b] = validity_map
                output_images[b] = output_image

            validity_maps_arr[i] = validity_maps
            images_arr[i] = images

        return images_arr, validity_maps_arr

    def reverse_resize_and_pad(self,
                               images_arr,
                               do_resize_and_pad,
                               resize_shape,
                               padding,
                               scale,
                               do_scale_reverse=True,
                               padding_modes=['constant'],
                               padding_value=0,
                               interpolation_modes=[Image.NEAREST]):
        '''
        reversing Resize and pad images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_resize_and_pad : bool
                N booleans to determine if image will be resized and padded
            resize_shape : list[int, int]
                height and width to resize
            padding : list[int, int, int, int]
                list of padding for left, top, right, bottom sides
            scale: list[float]
                scale used to re-scale the values of images
            do_scale_reverse : bool
                if set, then re-scale the values of images by scale
            padding_modes : list[str]
                list of modes for padding: 'constant', 'edge', 'reflect' or 'symmetric'
            padding_value : float
                value to pad with
            interpolation_modes : list[int]
                list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
            list[torch.Tensor] : list of N x 1 x H x W validity maps
        '''

        n_images_arr = len(images_arr)

        if len(padding_modes) < n_images_arr:
            padding_modes = \
                padding_modes + [padding_modes[-1]] * (n_images_arr - len(padding_modes))

        if len(interpolation_modes) < n_images_arr:
            interpolation_modes = \
                interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))

        for i, (images, interpolation_mode, padding_mode) in enumerate(zip(images_arr, interpolation_modes, padding_modes)):

            for b, _ in enumerate(images):
                image = images[b, ...]
                if do_resize_and_pad[b]:

                    r_height = resize_shape[0][b]
                    r_width = resize_shape[1][b]
                    pad_top = padding[0][b]
                    pad_bottom = padding[1][b]
                    pad_left = padding[2][b]
                    pad_right = padding[3][b]
                    n_height, n_width = image.shape[-2:]

                    if do_scale_reverse:
                       image = image * scale[b]

                    # padding performed
                    if r_height < n_height or r_width < n_width:
                        start_x = pad_left
                        start_y = pad_top
                        end_x = n_width - pad_right
                        end_y = n_height - pad_bottom

                        # extracting the resized image and resizing back to the original shape
                        image = image[..., start_y:end_y, start_x:end_x]
                        image = functional.resize(
                            image,
                            size=(n_height, n_width),
                            interpolation=interpolation_mode)

                    # padding not performed, center cropped instead
                    else:
                        # resizing the image and pad
                        image = functional.resize(
                            image,
                            size=(r_height, r_width),
                            interpolation=interpolation_mode)

                        new_pad_left = (n_width - r_width) // 2
                        new_pad_right = pad_left
                        new_pad_top = (n_height - r_height) // 2
                        new_pad_bottom = pad_top

                        image = functional.pad(
                            image,
                            (new_pad_left, new_pad_top, new_pad_right, new_pad_bottom),
                            fill=padding_value,
                            padding_mode=padding_mode)

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    '''
    Utility functions
    '''
    def scale_depth_map(self, images_arr, transform_performed):
        '''
        Scales depth maps according to transformations performed.
        '''
        if 'random_resize_and_crop' in transform_performed:
            do_resize, _, _, _, scale = transform_performed['random_resize_and_crop']
        elif 'random_resize_and_pad' in transform_performed:
            do_resize, _, _, scale = transform_performed['random_resize_and_pad']
        else:
            return images_arr

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):

                if do_resize[b]:
                    image = torch.where(image > 0, image / scale[b], image)
                    images[b] = image

            images_arr[i] = images

        return images_arr

    def adjust_intrinsics(self,
                          intrinsics_arr,
                          x_scales=1.0,
                          y_scales=1.0,
                          x_offsets=0.0,
                          y_offsets=0.0):
        '''
        Adjust the each camera intrinsics based on the provided scaling factors and offsets

        Arg(s):
            intrinsics : torch.Tensor[float32]
                3 x 3 camera intrinsics
            x_scales : list[float]
                scaling factor for x-direction focal lengths and optical centers
            y_scales : list[float]
                scaling factor for y-direction focal lengths and optical centers
            x_offsets : list[float]
                amount of horizontal offset to SUBTRACT from optical center
            y_offsets : list[float]
                amount of vertical offset to SUBTRACT from optical center
        Returns:
            torch.Tensor[float32] : 3 x 3 adjusted camera intrinsics
        '''

        for i, intrinsics in enumerate(intrinsics_arr):

            length = len(intrinsics)

            x_scales = [x_scales] * length if isinstance(x_scales, float) else x_scales
            y_scales = [y_scales] * length if isinstance(y_scales, float) else y_scales

            x_offsets = [x_offsets] * length if isinstance(x_offsets, float) else x_offsets
            y_offsets = [y_offsets] * length if isinstance(y_offsets, float) else y_offsets

            for b, K in enumerate(intrinsics):
                x_scale = x_scales[b]
                y_scale = y_scales[b]
                x_offset = x_offsets[b]
                y_offset = y_offsets[b]

                # Scale and subtract offset
                K[0, 0] = K[0, 0] * x_scale
                K[0, 2] = K[0, 2] * x_scale - x_offset
                K[1, 1] = K[1, 1] * y_scale
                K[1, 2] = K[1, 2] * y_scale - y_offset

            intrinsics[b] = K

        return intrinsics_arr

    def map_interpolation_mode_names_to_enums(self, interpolation_mode_names):
        '''
        Maps interpolation mode names to Image/InterpolationMode enums

        Arg(s):
            interpolation_mode_names : list[str]
                list of interpolation modes in string form
        Returns:
            list[int] : integers corresponding to Image/InterpolationMode enums
        '''

        n_name = len(interpolation_mode_names)

        interpolation_mode_enums = [None] * n_name

        for idx in range(n_name):
            interpolation_mode_enums[idx] = \
                self.__interpolation_mode_map[interpolation_mode_names[idx]]

        return interpolation_mode_enums
