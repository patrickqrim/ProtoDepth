import os, sys, glob, argparse, cv2, random, json, shutil
import numpy as np
import multiprocessing as mp
from PIL import Image
sys.path.insert(0, './')
import utils.src.data_utils as data_utils


# Set random seed
np.random.seed(42)
random.seed(42)

'''
Paths for SYNTHIA dataset
'''
SYNTHIA_DATA_DIRPATH = os.path.join('data', 'synthia')
SYNTHIA_TRAIN_DIRPATH = os.path.join(SYNTHIA_DATA_DIRPATH, 'train')
SYNTHIA_TEST_DIRPATH = os.path.join(SYNTHIA_DATA_DIRPATH, 'test')

N_TEST_SPARSE_DEPTH_SAMPLE = 5
N_TEST_SAMPLE = 4000

'''
Paths for KITTI depth completion
'''
KITTI_DEPTH_COMPLETION_DIRPATH = os.path.join('data', 'kitti_depth_completion')

KITTI_TRAIN_SPARSE_DEPTH_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'train_val_split', 'sparse_depth', 'train')

# To be concatenated to sequence path
KITTI_SPARSE_DEPTH_REFPATH = os.path.join('proj_depth', 'velodyne_raw')


'''
Output paths
'''
SYNTHIA_DEPTH_COMPLETION_DERIVED_DIRPATH = os.path.join(
    'data', 'synthia_derived-kitti')

TRAIN_SUPERVISED_REF_DIRPATH = os.path.join('training', 'synthia-kitti', 'supervised')
TRAIN_UNSUPERVISED_REF_DIRPATH = os.path.join('training', 'synthia-kitti', 'unsupervised')
VAL_REF_DIRPATH = os.path.join('validation', 'synthia-kitti')
TEST_REF_DIRPATH = os.path.join('testing', 'synthia-kitti')

# Paths to files for supervised training
TRAIN_SUPERVISED_IMAGE_FILEPATH = os.path.join(
    TRAIN_SUPERVISED_REF_DIRPATH, 'synthia_train_image.txt')
TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH = os.path.join(
    TRAIN_SUPERVISED_REF_DIRPATH, 'synthia_train_sparse_depth.txt')
TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_SUPERVISED_REF_DIRPATH, 'synthia_train_ground_truth.txt')
TRAIN_SUPERVISED_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_SUPERVISED_REF_DIRPATH, 'synthia_train_intrinsics.txt')

# Paths to files for unsupervised training
TRAIN_UNSUPERVISED_IMAGES_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'synthia_train_images.txt')
TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'synthia_train_sparse_depth.txt')
TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'synthia_train_ground_truth.txt')
TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'synthia_train_intrinsics.txt')

# Paths to files for testing
TEST_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'synthia_test_image.txt')
TEST_SPARSE_DEPTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'synthia_test_sparse_depth.txt')
TEST_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'synthia_test_ground_truth.txt')
TEST_INTRINSICS_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'synthia_test_intrinsics.txt')


def process_frame(args):
    '''
    Processes one frame

    Arg(s):
        args : tuple[str, str, str, str, list[str], list[str], bool]
            synthia image path at time t,
            synthia image path at time t-1,
            synthia image path at time t+1,
            synthia intrinsics path at t,
            synthia intrinsics path at t-1,
            synthia intrinsics path at t+1,
            synthia ground truth path at time t,
            kitti_sparse_depth_paths,
            output_dirpaths,
            save_image_triplet,
            sparse_depth_subsample_rate,
            paths_only
    Returns:
        list[str]: paths to synthia image
        list[str]: paths to synthia image triplets
        list[str]: paths to synthia sparse depth
        list[str]: paths to synthia intrinsics matrix
        list[str]: paths to synthia ground truth
    '''

    synthia_image_curr_path, \
        synthia_image_prev_path, \
        synthia_image_next_path, \
        synthia_info_curr_path, \
        synthia_info_prev_path, \
        synthia_info_next_path, \
        synthia_ground_truth_path, \
        kitti_sparse_depth_paths, \
        output_dirpaths, \
        save_image_triplet, \
        sparse_depth_subsample_rate, \
        paths_only = args

    # Get filename
    basename = os.path.basename(synthia_image_curr_path)
    filename = os.path.splitext(basename)[0]

    synthia_ground_truth = \
        np.array(Image.open(synthia_ground_truth_path).convert('RGB'), np.float32)

    # Conversion from RGB image to depth based on Synthia README
    # depth = 5000 * (R + G*256 + B*256*256) / (256*256*256 - 1)
    synthia_ground_truth = 5000 * \
        (synthia_ground_truth[..., 0] + 256 * synthia_ground_truth[..., 1] + 256 ** 2 * synthia_ground_truth[..., 2]) / \
        (256 ** 3 - 1)

    n_sample = len(kitti_sparse_depth_paths)

    synthia_image_dirpath, \
        synthia_images_dirpath, \
        synthia_sparse_depth_dirpath, \
        synthia_intrinsics_dirpath, \
        synthia_ground_truth_dirpath = output_dirpaths

    # Save intrinsics and duplicate paths
    with open(synthia_info_curr_path, 'rb') as f:
        synthia_info_curr = json.load(f)

    # Create 3 x 3 intrinsics matrix
    intrinsics = np.reshape(np.array(synthia_info_curr['intrinsic']['matrix']), (4, 4))

    offset_x = intrinsics[0, 3]
    offset_y = intrinsics[1, 3]
    focal_length = intrinsics[2, 3]

    intrinsics = np.eye(3)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length
    intrinsics[0, 2] = offset_x
    intrinsics[1, 2] = offset_y

    synthia_intrinsics_path = os.path.join(synthia_intrinsics_dirpath, filename + '.npy')

    if not paths_only:
        np.save(synthia_intrinsics_path, intrinsics)

    synthia_intrinsics_paths = [synthia_intrinsics_path] * n_sample

    # Check for static images
    if synthia_info_prev_path is not None and synthia_info_next_path is not None:

        # Load 4 x 4 world to camera pose matrices
        world_to_camera_curr = \
            np.reshape(np.array(synthia_info_curr['extrinsic']['matrix']), (4, 4))

        with open(synthia_info_prev_path, 'rb') as f:
            synthia_info_prev = json.load(f)

        with open(synthia_info_next_path, 'rb') as f:
            synthia_info_next = json.load(f)

        world_to_camera_prev = \
            np.reshape(np.array(synthia_info_prev['extrinsic']['matrix']), (4, 4))
        world_to_camera_next = \
            np.reshape(np.array(synthia_info_next['extrinsic']['matrix']), (4, 4))

        # Get parallax for prev to curr
        camera_to_world_prev = np.linalg.inv(world_to_camera_prev)
        prev_to_curr = np.matmul(camera_to_world_prev, world_to_camera_curr)
        t_prev_to_curr = np.linalg.norm(prev_to_curr[:3, 3])

        # Get parallax for next to curr
        camera_to_world_next = np.linalg.inv(world_to_camera_next)
        next_to_curr = np.matmul(camera_to_world_next, world_to_camera_curr)
        t_next_to_curr = np.linalg.norm(next_to_curr[:3, 3])

        # Check if we violate parallax of 50mm
        is_static = t_prev_to_curr < 0.05 or t_next_to_curr < 0.05
    else:
        is_static = None

    # Create image and image triplets
    synthia_image_paths = \
        [os.path.join(synthia_image_dirpath, basename)] * n_sample

    imagec = None

    if save_image_triplet and not is_static:
        synthia_images_paths = \
            [os.path.join(synthia_images_dirpath, basename)] * n_sample

        # Read images and concatenate together
        image_curr = cv2.imread(synthia_image_curr_path)
        image_prev = cv2.imread(synthia_image_prev_path)
        image_next = cv2.imread(synthia_image_next_path)

        imagec = np.concatenate([image_prev, image_curr, image_next], axis=1)

    else:
        synthia_images_paths = None

    # Set up sparse depth and ground truth paths
    synthia_sparse_depth_paths = []
    synthia_ground_truth_paths = \
        [os.path.join(synthia_ground_truth_dirpath, filename + '.png')] * n_sample

    # Get height and width of synthia image for rescaling
    n_height_synthia, n_width_synthia = synthia_ground_truth.shape

    # Define vectorize function to convert list to vector for indexing
    int_vector = np.vectorize(np.int_)

    for idx, kitti_sparse_depth_path in enumerate(kitti_sparse_depth_paths):

        if not paths_only:
            synthia_validity_map = np.zeros_like(synthia_ground_truth, dtype=np.float32)

            # Load KITTI validity map
            kitti_sparse_depth = data_utils.load_depth(kitti_sparse_depth_path)
            kitti_validity_map = np.where(
                kitti_sparse_depth > 0.0,
                1.0,
                0.0)

            n_height_kitti, n_width_kitti = kitti_validity_map.shape

            # Get xy positions of points from lidar depth map and get validity map
            y_nonzero, x_nonzero = np.nonzero(kitti_validity_map)

            n_point = np.prod(x_nonzero.shape)

            idx_selected = np.random.permutation(range(n_point))[:n_point//sparse_depth_subsample_rate]

            x_nonzero = x_nonzero[idx_selected]
            y_nonzero = y_nonzero[idx_selected]

            x_nonzero_indices = int_vector((x_nonzero / n_width_kitti) * n_width_synthia)
            y_nonzero_indices = int_vector((y_nonzero / n_height_kitti) * n_height_synthia)

            synthia_validity_map[y_nonzero_indices, x_nonzero_indices] = 1.0

            synthia_sparse_depth = synthia_ground_truth * synthia_validity_map

        # Append KITTI sequence and filename to VKIITTI filename
        kitti_path_parts = kitti_sparse_depth_path.split(os.sep)
        output_filename = filename + '-{}'.format(kitti_path_parts[-5] + '_' + kitti_path_parts[-1])

        # Store output paths
        synthia_sparse_depth_paths.append(os.path.join(synthia_sparse_depth_dirpath, output_filename))

        # Save to as PNG to disk
        if not paths_only:
            data_utils.save_depth(synthia_sparse_depth, synthia_sparse_depth_paths[-1])

        # Only save groundtruth depth once
        if idx == 0 and not paths_only:
            data_utils.save_depth(synthia_ground_truth, synthia_ground_truth_paths[-1])
            shutil.copy(synthia_image_curr_path, synthia_image_paths[-1])

            if imagec is not None:
                cv2.imwrite(synthia_images_paths[-1], imagec)

    return (synthia_image_paths,
            synthia_images_paths,
            synthia_sparse_depth_paths,
            synthia_intrinsics_paths,
            synthia_ground_truth_paths)


def setup_dataset_synthia(n_sparse_depth_sample_per_image=50,
                          sparse_depth_subsample_rate=4,
                          weather_conditions=['0', '2'],
                          paths_only=False,
                          n_thread=8):
    '''
    Fetch image, sparse depth, and ground truth paths for training

    Arg(s):
        n_sparse_depth__sample_per_image : int
            number of sparse depth sample to sample from KITTI for each image
        sparse_depth_subsample_rate : int
            rate of sparse depth points subsampling
        weather_conditions : list[int]
            weather conditions that will be usd in dataset
        paths_only : bool
            if set, then only produces paths
        n_thread : int
            number of threads to use for multiprocessing
    '''

    # Paths for supervised training
    train_supervised_image_paths = []
    train_supervised_sparse_depth_paths = []
    train_supervised_ground_truth_paths = []
    train_supervised_intrinsics_paths = []

    # Paths for unsupervised training
    train_unsupervised_images_paths = []
    train_unsupervised_sparse_depth_paths = []
    train_unsupervised_ground_truth_paths = []
    train_unsupervised_intrinsics_paths = []

    # Paths for testing
    test_image_paths = []
    test_sparse_depth_paths = []
    test_ground_truth_paths = []
    test_intrinsics_paths = []

    # Get all KITTI sparse depth paths
    kitti_sparse_depth_paths = glob.glob(
        os.path.join(KITTI_TRAIN_SPARSE_DEPTH_DIRPATH, '*', KITTI_SPARSE_DEPTH_REFPATH, '*', '*.png'))

    kitti_sparse_depth_paths = sorted(kitti_sparse_depth_paths)

    # Iterate through train and test directories in Synthia
    for data_split in ['train', 'test']:

        if data_split == 'train':
            refdir = SYNTHIA_TRAIN_DIRPATH
        else:
            refdir = SYNTHIA_TEST_DIRPATH

        sequence_dirpaths = sorted(glob.glob(os.path.join(refdir, '*/', '*/')))

        # Iterate through sequences:
        # Example: data/synthia/train/test5_10segs_weather_0_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None/19-10-2018_12-47-37
        for sequence_dirpath in sequence_dirpaths:

            sequence_dirpath_parts = sequence_dirpath.split('_')
            weather_condition_idx = sequence_dirpath_parts.index('weather')
            weather_condition = sequence_dirpath_parts[weather_condition_idx+1]

            if weather_condition not in weather_conditions:
                continue

            print('Processing Synthia sequence: {}'.format(sequence_dirpath))

            # Construct output directory for image
            # Example: data/synthia_kitti_derived/train/test5_10segs_weather_0_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None/19-10-2018_12-47-37/rgb
            output_sequence_image_dirpath = \
                os.path.join(sequence_dirpath, 'rgb').replace(SYNTHIA_DATA_DIRPATH, SYNTHIA_DEPTH_COMPLETION_DERIVED_DIRPATH)

            output_image_dirpath = os.path.join(
                output_sequence_image_dirpath, 'image')
            output_images_dirpath = os.path.join(
                output_sequence_image_dirpath, 'images')

            # Construct output directory for depth
            # Example: data/synthia_kitti_derived/train/test5_10segs_weather_0_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None/19-10-2018_12-47-37/depth
            output_sequence_depth_dirpath = \
                os.path.join(sequence_dirpath, 'depth').replace(SYNTHIA_DATA_DIRPATH, SYNTHIA_DEPTH_COMPLETION_DERIVED_DIRPATH)

            output_sparse_depth_dirpath = os.path.join(
                output_sequence_depth_dirpath, 'sparse_depth')
            output_ground_truth_dirpath = os.path.join(
                output_sequence_depth_dirpath, 'ground_truth')

            # Construct output directory for intrinsics
            # Example: data/synthia_kitti_derived/train/test5_10segs_weather_0_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None/19-10-2018_12-47-37/intrinsics
            output_intrinsics_dirpath = \
                os.path.join(sequence_dirpath, 'intrinsics').replace(SYNTHIA_DATA_DIRPATH, SYNTHIA_DEPTH_COMPLETION_DERIVED_DIRPATH)

            # Create output directories
            output_dirpaths = [
                output_image_dirpath,
                output_images_dirpath,
                output_sparse_depth_dirpath,
                output_intrinsics_dirpath,
                output_ground_truth_dirpath
            ]

            for output_dirpath in output_dirpaths:
                if not os.path.exists(output_dirpath):
                    os.makedirs(output_dirpath)

            # Fetch image paths
            synthia_image_paths = \
                sorted(glob.glob(os.path.join(sequence_dirpath, 'RGB', '*.png')))

            # Fetch info paths
            synthia_info_paths = \
                sorted(glob.glob(os.path.join(sequence_dirpath, 'Information', '*.json')))

            # Fetch ground_truth depth paths
            synthia_ground_truth_paths = \
                sorted(glob.glob(os.path.join(sequence_dirpath, 'Depth', '*.png')))

            n_sample = len(synthia_image_paths)

            # Check that data streams are aligned
            assert n_sample == len(synthia_ground_truth_paths)
            assert n_sample == len(synthia_info_paths)

            data = zip(
                synthia_image_paths,
                synthia_info_paths,
                synthia_ground_truth_paths)

            pool_inputs = []
            n_total = 0

            for idx, datum in enumerate(data):

                synthia_image_curr_path, synthia_info_curr_path, synthia_ground_truth_path = datum

                synthia_image_curr_filename = \
                    os.path.splitext(os.path.basename(synthia_image_curr_path))[0]
                synthia_info_curr_filename = \
                    os.path.splitext(os.path.basename(synthia_info_curr_path))[0]
                synthia_ground_truth_filename = \
                    os.path.splitext(os.path.basename(synthia_ground_truth_path))[0]

                assert synthia_image_curr_filename == synthia_info_curr_filename
                assert synthia_image_curr_filename == synthia_ground_truth_filename

                if idx in range(1, n_sample - 1) and data_split == 'train':
                    synthia_image_prev_path = synthia_image_paths[idx-1]
                    synthia_image_next_path = synthia_image_paths[idx+1]
                    synthia_info_prev_path = synthia_info_paths[idx-1]
                    synthia_info_next_path = synthia_info_paths[idx+1]
                    save_image_triplet = True
                else:
                    synthia_image_prev_path = None
                    synthia_image_next_path = None
                    synthia_info_prev_path = None
                    synthia_info_next_path = None
                    save_image_triplet = False

                # Shuffle the KITTI samples and choose N of them
                kitti_subset_sparse_depth_paths = \
                    np.random.permutation(kitti_sparse_depth_paths)[0:n_sparse_depth_sample_per_image]

                pool_inputs.append((
                    synthia_image_curr_path,
                    synthia_image_prev_path,
                    synthia_image_next_path,
                    synthia_info_curr_path,
                    synthia_info_prev_path,
                    synthia_info_next_path,
                    synthia_ground_truth_path,
                    kitti_subset_sparse_depth_paths,
                    output_dirpaths,
                    save_image_triplet,
                    sparse_depth_subsample_rate,
                    paths_only))

            with mp.Pool(n_thread) as pool:
                pool_results = pool.map(process_frame, pool_inputs)

            for result in pool_results:

                image_paths, \
                    images_paths, \
                    sparse_depth_paths, \
                    intrinsics_paths, \
                    ground_truth_paths = result

                if data_split == 'train':
                    # If we have image triplet
                    if images_paths is not None:
                        train_unsupervised_images_paths.extend(images_paths)
                        train_unsupervised_sparse_depth_paths.extend(sparse_depth_paths)
                        train_unsupervised_intrinsics_paths.extend(intrinsics_paths)
                        train_unsupervised_ground_truth_paths.extend(ground_truth_paths)

                    # Append all images for supervised training
                    train_supervised_image_paths.extend(image_paths)
                    train_supervised_sparse_depth_paths.extend(sparse_depth_paths)
                    train_supervised_ground_truth_paths.extend(ground_truth_paths)
                    train_supervised_intrinsics_paths.extend(intrinsics_paths)
                else:
                    test_image_paths.extend(image_paths[0:min(N_TEST_SPARSE_DEPTH_SAMPLE, n_sparse_depth_sample_per_image)])
                    test_sparse_depth_paths.extend(sparse_depth_paths[0:min(N_TEST_SPARSE_DEPTH_SAMPLE, n_sparse_depth_sample_per_image)])
                    test_ground_truth_paths.extend(ground_truth_paths[0:min(N_TEST_SPARSE_DEPTH_SAMPLE, n_sparse_depth_sample_per_image)])
                    test_intrinsics_paths.extend(intrinsics_paths[0:min(N_TEST_SPARSE_DEPTH_SAMPLE, n_sparse_depth_sample_per_image)])

                n_total = n_total + len(image_paths)

            print('Generated {} total samples for {}'.format(n_total, sequence_dirpath))

    '''
    Write supervised training paths to disk
    '''
    print('Writing {} supervised training image paths to {}'.format(
        len(train_supervised_image_paths),
        TRAIN_SUPERVISED_IMAGE_FILEPATH))
    data_utils.write_paths(
        TRAIN_SUPERVISED_IMAGE_FILEPATH,
        train_supervised_image_paths)

    print('Writing {} supervised training sparse depth paths to {}'.format(
        len(train_supervised_sparse_depth_paths),
        TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH))
    data_utils.write_paths(
        TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH,
        train_supervised_sparse_depth_paths)

    print('Writing {} supervised training intrinsics paths to {}'.format(
        len(train_supervised_intrinsics_paths),
        TRAIN_SUPERVISED_INTRINSICS_FILEPATH))
    data_utils.write_paths(
        TRAIN_SUPERVISED_INTRINSICS_FILEPATH,
        train_supervised_intrinsics_paths)

    print('Writing {} supervised training groundtruth depth paths to {}'.format(
        len(train_supervised_ground_truth_paths),
        TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH))
    data_utils.write_paths(
        TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH,
        train_supervised_ground_truth_paths)

    '''
    Write unsupervised training paths to disk
    '''
    print('Writing {} unsupervised training image paths to {}'.format(
        len(train_unsupervised_images_paths),
        TRAIN_UNSUPERVISED_IMAGES_FILEPATH))
    data_utils.write_paths(
        TRAIN_UNSUPERVISED_IMAGES_FILEPATH,
        train_unsupervised_images_paths)

    print('Writing {} unsupervised training sparse depth paths to {}'.format(
        len(train_unsupervised_sparse_depth_paths),
        TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH))
    data_utils.write_paths(
        TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH,
        train_unsupervised_sparse_depth_paths)

    print('Writing {} unsupervised training intrinsics paths to {}'.format(
        len(train_unsupervised_intrinsics_paths),
        TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH))
    data_utils.write_paths(
        TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH,
        train_unsupervised_intrinsics_paths)

    print('Writing {} unsupervised training groundtruth depth paths to {}'.format(
        len(train_unsupervised_ground_truth_paths),
        TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH))
    data_utils.write_paths(
        TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH,
        train_unsupervised_ground_truth_paths)

    '''
    Write testing paths to disk
    '''
    # Subsample from testing set
    idx_selected = np.random.permutation(range(len(test_image_paths)))[0:N_TEST_SAMPLE]
    test_image_paths = np.array(test_image_paths)[idx_selected]
    test_image_paths = test_image_paths.tolist()

    test_sparse_depth_paths = np.array(test_sparse_depth_paths)[idx_selected]
    test_sparse_depth_paths = test_sparse_depth_paths.tolist()

    test_intrinsics_paths = np.array(test_intrinsics_paths)[idx_selected]
    test_intrinsics_paths = test_intrinsics_paths.tolist()

    test_ground_truth_paths = np.array(test_ground_truth_paths)[idx_selected]
    test_ground_truth_paths = test_ground_truth_paths.tolist()

    print('Writing {} testing image paths to {}'.format(
        len(test_image_paths),
        TEST_IMAGE_FILEPATH))
    data_utils.write_paths(
        TEST_IMAGE_FILEPATH,
        test_image_paths)

    print('Writing {} testing sparse depth paths to {}'.format(
        len(test_sparse_depth_paths),
        TEST_SPARSE_DEPTH_FILEPATH))
    data_utils.write_paths(
        TEST_SPARSE_DEPTH_FILEPATH,
        test_sparse_depth_paths)

    print('Writing {} testing intrinsics paths to {}'.format(
        len(test_intrinsics_paths),
        TEST_INTRINSICS_FILEPATH))
    data_utils.write_paths(
        TEST_INTRINSICS_FILEPATH,
        test_intrinsics_paths)

    print('Writing {} testing groundtruth depth paths to {}'.format(
        len(test_ground_truth_paths),
        TEST_GROUND_TRUTH_FILEPATH))
    data_utils.write_paths(
        TEST_GROUND_TRUTH_FILEPATH,
        test_ground_truth_paths)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_sparse_depth_sample_per_image',
        type=int, default=50, help='Number of KITTI sparse depth map to sample for each image')
    parser.add_argument('--sparse_depth_subsample_rate',
        type=int, default=4, help='Rate of sparse depth points subsampling')
    parser.add_argument('--weather_conditions',
        nargs='+', type=str, default=['0', '2'], help='Weather conditions: 0, 2, 3, 4, 5')
    parser.add_argument('--paths_only',
        action='store_true', help='If set, then generate paths only')
    parser.add_argument('--n_thread',
        type=int, default=8, help='Number of threads to use for multiprocessing')

    args = parser.parse_args()

    dirpaths = [
        TRAIN_SUPERVISED_REF_DIRPATH,
        TRAIN_UNSUPERVISED_REF_DIRPATH,
        VAL_REF_DIRPATH,
        TEST_REF_DIRPATH
    ]

    # Create directories for output files
    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    # Set up dataset
    setup_dataset_synthia(
        n_sparse_depth_sample_per_image=args.n_sparse_depth_sample_per_image,
        sparse_depth_subsample_rate=args.sparse_depth_subsample_rate,
        weather_conditions=args.weather_conditions,
        paths_only=args.paths_only,
        n_thread=args.n_thread)
