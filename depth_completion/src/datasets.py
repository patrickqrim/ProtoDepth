import torch
import numpy as np
from utils.src import data_utils


def load_image_triplet(path, normalize=True, data_format='CHW'):
    '''
    Load in triplet frames from path

    Arg(s):
        path : str
            path to image triplet
        normalize : bool
            if set, normalize to [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : image at t - 1
        numpy[float32] : image at t
        numpy[float32] : image at t + 1
    '''

    # Load image triplet and split into images at t-1, t, t+1
    images = data_utils.load_image(
        path,
        normalize=normalize,
        data_format=data_format)

    # Split along width
    image1, image0, image2 = np.split(images, indices_or_sections=3, axis=-1)

    return image1, image0, image2

def load_depth(depth_path, data_format='CHW'):
    '''
    Load depth

    Arg(s):
        depth_path : str
            path to depth map
        data_format : str
            'CHW', or 'HWC'
    Return:
        numpy[float32] : depth map (1 x H x W)
    '''

    return data_utils.load_depth(depth_path, data_format=data_format)

def load_depth_with_validity_map(depth_path, data_format='CHW'):
    '''
    Load depth and validity maps


    Arg(s):
        depth_path : str
            path to depth map
        data_format : str
            'CHW', or 'HWC'
        Returns:
            numpy[float32] : depth and validity map (2 x H x W)
    '''

    depth_map, validity_map = data_utils.load_depth_with_validity_map(
        depth_path,
        data_format=data_format)

    if data_format == 'CHW':
        return np.concatenate([depth_map, validity_map], axis=0)
    elif data_format == 'HWC':
        return np.concatenate([depth_map, validity_map], axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width+1)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height+1)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    # Adjust intrinsics
    if intrinsics is not None:
        offset_principal_point = [[0.0, 0.0, -x_start],
                                  [0.0, 0.0, -y_start],
                                  [0.0, 0.0, 0.0     ]]

        intrinsics = [
            in_ + offset_principal_point for in_ in intrinsics
        ]

        return outputs, intrinsics
    else:
        return outputs


class DepthCompletionMonocularTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) camera image at t
        (2) left camera image at t - 1
        (3) left camera image at t + 1
        (4) sparse depth map at t
        (5) intrinsic camera calibration matrix

    Arg(s):
        images_paths : list[str]
            paths to camera images t-1, t, t+1
        sparse_depth_paths : list[str]
            paths to sparse depth maps at t
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        dataset_uid : str
            unique dataset identifier
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 images_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 dataset_uid,
                 random_crop_shape=None,
                 random_crop_type=None):

        self.n_sample = len(images_paths)

        input_paths = [
            sparse_depth_paths,
            intrinsics_paths
        ]

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.images_paths = images_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.dataset_uid = dataset_uid

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image1, image0, image2 = load_image_triplet(
            self.images_paths[index],
            normalize=False)

        # Load sparse depth
        sparse_depth0 = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image0, image1, image2, sparse_depth0], [intrinsics] = random_crop(
                inputs=[image0, image1, image2, sparse_depth0],
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert to float32
        image0, image1, image2, sparse_depth0, intrinsics = [
            T.astype(np.float32)
            for T in [image0, image1, image2, sparse_depth0, intrinsics]
        ]

        return image0, image1, image2, sparse_depth0, intrinsics

    def __len__(self):
        return self.n_sample


class DepthCompletionMonocularDistillationTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) camera image at t
        (2) left camera image at t - 1
        (3) left camera image at t + 1
        (4) sparse depth map at t
        (5) intrinsic camera calibration matrix
        (6) teacher depth predictions at time t

    Arg(s):
        images_paths : list[str]
            paths to camera images t-1, t, t+1
        sparse_depth_paths : list[str]
            paths to sparse depth maps at t
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        teacher_ensemble_depth_paths : list[list[str]]
            list of list of paths to teacher depth predictions at t
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 images_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 teacher_ensemble_depth_paths,
                 random_crop_shape=None,
                 random_crop_type=None):

        self.n_sample = len(images_paths)

        input_paths = [
            sparse_depth_paths,
            intrinsics_paths
        ]

        input_paths = input_paths + teacher_ensemble_depth_paths

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.images_paths = images_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths
        self.teacher_ensemble_depth_paths = teacher_ensemble_depth_paths

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image1, image0, image2 = load_image_triplet(
            self.images_paths[index],
            normalize=False)

        # Load sparse depth
        sparse_depth0 = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        # Load teacher ensemble predictions
        teacher_ensemble_depth0 = []

        for paths in self.teacher_ensemble_depth_paths:
            teacher_ensemble_depth0.append(
                data_utils.load_depth(
                    path=paths[index],
                    data_format=self.data_format))

        teacher_ensemble_depth0 = np.concatenate(teacher_ensemble_depth0, axis=0)

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image0, image1, image2, sparse_depth0, teacher_ensemble_depth0], [intrinsics] = random_crop(
                inputs=[image0, image1, image2, sparse_depth0, teacher_ensemble_depth0],
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert to float32
        image0, image1, image2, sparse_depth0, intrinsics, teacher_ensemble_depth0 = [
            T.astype(np.float32)
            for T in [image0, image1, image2, sparse_depth0, intrinsics, teacher_ensemble_depth0]
        ]

        return image0, image1, image2, sparse_depth0, intrinsics, teacher_ensemble_depth0

    def __len__(self):
        return self.n_sample


class DepthCompletionSupervisedTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) camera image at t
        (2) sparse depth map at t
        (3) intrinsic camera calibration matrix
        (4) ground truth depth map at t

    Arg(s):
        image_paths : list[str]
            paths to camera images t-1, t, t+1
        sparse_depth_paths : list[str]
            paths to sparse depth maps at t
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        ground_truth_paths : list[str]
            paths to ground truth depth maps at time t
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 ground_truth_paths,
                 random_crop_shape=None,
                 random_crop_type=None,
                 load_image_triplet=True):

        self.n_sample = len(image_paths)

        assert ground_truth_paths is not None and None not in ground_truth_paths

        input_paths = [
            sparse_depth_paths,
            intrinsics_paths,
            ground_truth_paths
        ]

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths
        self.ground_truth_paths = ground_truth_paths

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.load_image_triplet = load_image_triplet

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
            path=self.image_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        if self.intrinsics_paths[index] is not None:
            intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)
        else:
            intrinsics = np.eye(3)

        # Load ground truth
        ground_truth = data_utils.load_depth(
            path=self.ground_truth_paths[index],
            data_format=self.data_format)

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image, sparse_depth, ground_truth], [intrinsics] = random_crop(
                inputs=[image, sparse_depth, ground_truth],
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert to float32
        image, sparse_depth, intrinsics, ground_truth = [
            T.astype(np.float32)
            for T in [image, sparse_depth, intrinsics, ground_truth]
        ]

        return image, sparse_depth, intrinsics, ground_truth

    def __len__(self):
        return self.n_sample


class DepthCompletionInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) intrinsic camera calibration matrix

    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        dataset_uid : str
            unique dataset identifier
        ground_truth_paths : list[str]
            paths to ground truth depth maps
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 dataset_uid,
                 ground_truth_paths=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        for paths in [sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.is_available_ground_truth = \
           ground_truth_paths is not None and all([x is not None for x in ground_truth_paths])

        if self.is_available_ground_truth:
            self.ground_truth_paths = ground_truth_paths

        self.data_format = 'CHW'
        self.load_image_triplets = load_image_triplets

        self.dataset_uid = dataset_uid

    def __getitem__(self, index):

        # Load image
        if self.load_image_triplets:
            _, image, _ = load_image_triplet(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)
        else:
            image = data_utils.load_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index])

        inputs = [
            image,
            sparse_depth,
            intrinsics
        ]

        # Load ground truth if available
        if self.is_available_ground_truth:
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
            inputs.append(ground_truth)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        # Return image, sparse_depth, intrinsics, and if available, ground_truth
        return inputs

    def __len__(self):
        return self.n_sample
