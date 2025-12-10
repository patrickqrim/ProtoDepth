import argparse
import torch
from depth_completion import train


class ParseStrFloatKeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split("=")
                getattr(namespace, self.dest)[key] = float(value)
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))


parser = argparse.ArgumentParser()

# Training filepaths
parser.add_argument('--train_image_paths',
    nargs='+', type=str, required=True, help='Paths to list of training image triplets paths')
parser.add_argument('--train_sparse_depth_paths',
    nargs='+', type=str, required=True, help='Paths to list of training sparse depth paths')
parser.add_argument('--train_intrinsics_paths',
    nargs='+', type=str, default=None, help='Paths to list of training camera intrinsics paths')
parser.add_argument('--train_ground_truth_paths',
    nargs='+', type=str, default=None, help='Paths to list of training ground_truth paths')
parser.add_argument('--train_dataset_uids',
    nargs='+', type=str, default=None, help='Unique dataset ID string for each training dataset')

# Validation filepaths
parser.add_argument('--val_image_paths',
    nargs='+', type=str, default=None, help='Path to list of validation image paths')
parser.add_argument('--val_sparse_depth_paths',
    nargs='+', type=str, default=None, help='Path to list of validation sparse depth paths')
parser.add_argument('--val_intrinsics_paths',
    nargs='+', type=str, default=None, help='Path to list of validation camera intrinsics paths')
parser.add_argument('--val_ground_truth_paths',
    nargs='+', type=str, default=None, help='Path to list of validation ground truth depth paths')
parser.add_argument('--val_dataset_uids',
    nargs='+', type=str, default=None, help='Unique dataset ID string for each validation dataset')

# TokenCDC settings
parser.add_argument('--image_pool_size',
    type=int, default=10, help='Size of each image key/token pool')
parser.add_argument('--depth_pool_size',
    type=int, default=10, help='Size of each depth key/token pool')
parser.add_argument('--unfreeze_model',
    action='store_true', default=False, help='Use flag to freeze model and only learn tokens')
parser.add_argument('--domain_agnostic',
    action='store_true', default=False, help='Use flag to evaluate on domain-agnostic setting')

# Depth network settings
parser.add_argument('--model_name',
    type=str, default='kbnet_void', help='Depth completion model name')
parser.add_argument('--network_modules',
    nargs='+', type=str, default=[], help='modules to build for networks')
parser.add_argument('--min_predict_depth',
    type=float, default=0.10, help='Minimum value of predicted depth')
parser.add_argument('--max_predict_depth',
    type=float, default=10.00, help='Maximum value of predicted depth')

# Training settings
parser.add_argument('--train_batch_size',
    type=int, default=8, help='Number of samples per batch (divisible by number of datasets)')
parser.add_argument('--train_crop_shapes',
    nargs='+', type=int, default=[480, 640], help='List of (height, width) crop shapes for training data')
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[1e-4, 5e-5], help='Space delimited list of learning rates')
parser.add_argument('--pose_learning_rates',
    nargs='+', type=float, default=[], help='Space delimited list of PoseNet learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[5, 10], help='Space delimited list to change learning rate')

# Augmentation settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[1.00], help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[-1], help='If not -1, then space delimited list to change augmentation probability')

# Photometric data augmentations
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[-1, -1], help='Range of brightness adjustments for augmentation, if does not contain -1, apply random brightness')
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=[-1, -1], help='Range of contrast adjustments for augmentation, if does not contain -1, apply random contrast')
parser.add_argument('--augmentation_random_gamma',
    nargs='+', type=float, default=[-1, 1], help='Range of gamma adjustments for augmentation, if does not contain -1, apply random gamma')
parser.add_argument('--augmentation_random_hue',
    nargs='+', type=float, default=[-1, -1], help='Range of hue adjustments for augmentation, if does not contain -1, apply random hue')
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=[-1, -1], help='Range of saturation adjustments for augmentation, if does not contain -1, apply random saturation')
parser.add_argument('--augmentation_random_gaussian_blur_kernel_size',
    nargs='+', type=int, default=[-1, -1], help='List of kernel sizes to be used for gaussian blur')
parser.add_argument('--augmentation_random_gaussian_blur_sigma_range',
    nargs='+', type=float, default=[-1, -1], help='Min and max standard deviation for gaussian blur')
parser.add_argument('--augmentation_random_noise_type',
    type=str, default='none', help='Random noise to add: gaussian, uniform')
parser.add_argument('--augmentation_random_noise_spread',
    type=float, default=-1, help='If gaussian noise, then standard deviation; if uniform, then min-max range')

# Geometric data augmentations
parser.add_argument('--augmentation_padding_mode',
    type=str, default='constant', help='Padding used: constant, edge, reflect or symmetric.')
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=['none'], help='Random crop type for data augmentation: horizontal, vertical, anchored, bottom')
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'], help='Random flip type for data augmentation: horizontal, vertical')
parser.add_argument('--augmentation_random_rotate_max',
    type=float, default=-1, help='Max angle for random rotation, disabled if -1')
parser.add_argument('--augmentation_random_crop_and_pad',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to crop and pad')
parser.add_argument('--augmentation_random_resize_to_shape',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to resize')
parser.add_argument('--augmentation_random_resize_and_pad',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to resize and pad (or crop if max size is larger than 1)')
parser.add_argument('--augmentation_random_resize_and_crop',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to resize and crop ')

# Occlusion data augmentations
parser.add_argument('--augmentation_random_remove_patch_percent_range_image',
    nargs='+', type=float, default=[-1, -1], help='If not -1, randomly remove patches covering percentage of image as augmentation')
parser.add_argument('--augmentation_random_remove_patch_size_image',
    nargs='+', type=int, default=[-1, -1], help='If not -1, patch size for random remove patch augmentation for image')
parser.add_argument('--augmentation_random_remove_patch_percent_range_depth',
    nargs='+', type=float, default=[-1, -1], help='If not -1, randomly remove patches covering percentage of depth map as augmentation')
parser.add_argument('--augmentation_random_remove_patch_size_depth',
    nargs='+', type=int, default=[-1, -1], help='If not -1, patch size for random remove patch augmentation for depth map')

# Loss function settings
parser.add_argument('--supervision_type',
    type=str, default='unsupervised', help='Supervision type for training')
parser.add_argument('--w_losses',
    nargs='+', type=str, action=ParseStrFloatKeyValueAction, help='Weight of each loss term as key-value pairs: w_color=0.90 w_smoothness=2.00')

parser.add_argument('--frozen_model_paths',
    nargs='+', type=str, default=[], help='Path to frozen models from checkpoint')

# Evaluation settings
parser.add_argument('--min_evaluate_depths',
    nargs='+', type=float, default=[0.20], help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depths',
    nargs='+', type=float, default=[5.00], help='Maximum value of depth to evaluate')
parser.add_argument('--evaluation_protocols',
    nargs='+', type=str, default=['default'], help='Protocol for evaluation i.e. vkitti, nuscenes, default')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=1000, help='Number of iterations for each checkpoint')
parser.add_argument('--n_step_per_summary',
    type=int, default=1000, help='Number of iterations before logging summary')
parser.add_argument('--n_image_per_summary',
    type=int, default=4, help='Number of samples to include in visual display summary')
parser.add_argument('--start_step_validation',
    type=int, default=5000, help='Number of steps before starting validation')
parser.add_argument('--restore_paths',
    nargs='+', type=str, default=[], help='Path to restore depth or pose model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    # Depth network settings
    args.model_name = args.model_name.lower()

    # Loss settings
    args.supervision_type = args.supervision_type.lower()

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)
    if len(args.pose_learning_rates) != len(args.learning_schedule):
        args.pose_learning_rates = args.learning_rates

    args.augmentation_random_crop_type = [
        crop_type.lower() for crop_type in args.augmentation_random_crop_type
    ]

    args.augmentation_random_flip_type = [
        flip_type.lower() for flip_type in args.augmentation_random_flip_type
    ]

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    train(
        # Training filepaths
        train_image_paths=args.train_image_paths,
        train_sparse_depth_paths=args.train_sparse_depth_paths,
        train_intrinsics_paths=args.train_intrinsics_paths,
        train_ground_truth_paths=args.train_ground_truth_paths,
        train_dataset_uids=args.train_dataset_uids,  # TokenCDC: Added train dataset IDs
        # Validation filepaths
        val_image_paths=args.val_image_paths,
        val_sparse_depth_paths=args.val_sparse_depth_paths,
        val_intrinsics_paths=args.val_intrinsics_paths,
        val_ground_truth_paths=args.val_ground_truth_paths,
        val_dataset_uids=args.val_dataset_uids,  # TokenCDC: Added val dataset IDs
        # TokenCDC-specific settings
        image_pool_size=args.image_pool_size,
        depth_pool_size=args.depth_pool_size,
        unfreeze_model=args.unfreeze_model,
        domain_agnostic=args.domain_agnostic,
        # Depth network settings
        model_name=args.model_name,
        network_modules=args.network_modules,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        # Training settings
        train_batch_size=args.train_batch_size,
        train_crop_shapes=args.train_crop_shapes,
        learning_rates=args.learning_rates,
        pose_learning_rates=args.pose_learning_rates,
        learning_schedule=args.learning_schedule,
        augmentation_probabilities=args.augmentation_probabilities,
        augmentation_schedule=args.augmentation_schedule,
        # Photometric data augmentations
        augmentation_random_brightness=args.augmentation_random_brightness,
        augmentation_random_contrast=args.augmentation_random_contrast,
        augmentation_random_gamma=args.augmentation_random_gamma,
        augmentation_random_hue=args.augmentation_random_hue,
        augmentation_random_saturation=args.augmentation_random_saturation,
        augmentation_random_gaussian_blur_kernel_size=args.augmentation_random_gaussian_blur_kernel_size,
        augmentation_random_gaussian_blur_sigma_range=args.augmentation_random_gaussian_blur_sigma_range,
        augmentation_random_noise_type=args.augmentation_random_noise_type,
        augmentation_random_noise_spread=args.augmentation_random_noise_spread,
        # Geometric data augmentations
        augmentation_padding_mode=args.augmentation_padding_mode,
        augmentation_random_crop_type=args.augmentation_random_crop_type,
        augmentation_random_flip_type=args.augmentation_random_flip_type,
        augmentation_random_rotate_max=args.augmentation_random_rotate_max,
        augmentation_random_crop_and_pad=args.augmentation_random_crop_and_pad,
        augmentation_random_resize_to_shape=args.augmentation_random_resize_to_shape,
        augmentation_random_resize_and_pad=args.augmentation_random_resize_and_pad,
        augmentation_random_resize_and_crop=args.augmentation_random_resize_and_crop,
        # Occlusion data augmentations
        augmentation_random_remove_patch_percent_range_image=args.augmentation_random_remove_patch_percent_range_image,
        augmentation_random_remove_patch_size_image=args.augmentation_random_remove_patch_size_image,
        augmentation_random_remove_patch_percent_range_depth=args.augmentation_random_remove_patch_percent_range_depth,
        augmentation_random_remove_patch_size_depth=args.augmentation_random_remove_patch_size_depth,
        # Loss function settings
        supervision_type=args.supervision_type,
        w_losses=args.w_losses,
        # Evaluation settings
        min_evaluate_depths=args.min_evaluate_depths,
        max_evaluate_depths=args.max_evaluate_depths,
        evaluation_protocols=args.evaluation_protocols,
        # Checkpoint settings
        checkpoint_path=args.checkpoint_path,
        n_step_per_checkpoint=args.n_step_per_checkpoint,
        n_step_per_summary=args.n_step_per_summary,
        n_image_per_summary=args.n_image_per_summary,
        start_step_validation=args.start_step_validation,
        restore_paths=args.restore_paths,
        # Hardware settings
        device=args.device,
        n_thread=args.n_thread)
