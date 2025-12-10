import argparse
import torch
from depth_completion import run


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--image_path',
    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--sparse_depth_path',
    type=str, required=True, help='Path to list of sparse depth paths')
parser.add_argument('--intrinsics_path',
    type=str, required=True, help='Path to list of camera intrinsics paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth depth paths')

# Restore path settings
parser.add_argument('--restore_paths',
    nargs='+', type=str, default=[], help='Path to restore depth or pose model from checkpoint')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=3, help='Number of input image channels')
parser.add_argument('--input_channels_depth',
    type=int, default=2, help='Number of input depth channels')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')

# Depth network settings
parser.add_argument('--model_name',
    type=str, default='kbnet_void', help='Depth completion model name')
parser.add_argument('--network_modules',
    nargs='+', type=str, default=[], help='modules to build for networks')
parser.add_argument('--min_predict_depth',
    type=float, default=0.10, help='Minimum value of predicted depth')
parser.add_argument('--max_predict_depth',
    type=float, default=10.00, help='Maximum value of predicted depth')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=[0.20], help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=[5.00], help='Maximum value of depth to evaluate')
parser.add_argument('--evaluation_protocol',
    type=str, default='default', help='Protocol for evaluation i.e. vkitti, nuscenes, default')

# Output settings
parser.add_argument('--output_path',
    type=str, required=True, help='Path to save outputs')
parser.add_argument('--save_outputs',
    action='store_true', help='If set then store inputs and outputs into output path')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep original input filenames')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')


args = parser.parse_args()

if __name__ == '__main__':

    # Depth network settings
    args.model_name = args.model_name.lower()

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    run(image_path=args.image_path,
        sparse_depth_path=args.sparse_depth_path,
        intrinsics_path=args.intrinsics_path,
        ground_truth_path=args.ground_truth_path,
        # Restore path settings
        restore_paths=args.restore_paths,
        # Input settings
        input_channels_image=args.input_channels_image,
        input_channels_depth=args.input_channels_depth,
        normalized_image_range=args.normalized_image_range,
        # Depth network settings
        model_name=args.model_name,
        network_modules=args.network_modules,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        # Evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        evaluation_protocol=args.evaluation_protocol,
        # Output settings
        output_path=args.output_path,
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        # Hardware settings
        device=args.device)
