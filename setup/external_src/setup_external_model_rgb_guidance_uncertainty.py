'''
Authors:
Tian Yu Liu <tianyu@cs.ucla.edu>
Parth Agrawal <parthagrawal24@ucla.edu>
Allison Chen <allisonchen2@ucla.edu>
Alex Wong <alex.wong@yale.edu>

If you use this code, please cite the following paper:
T.Y. Liu, P. Agrawal, A. Chen, B.W. Hong, and A. Wong. Monitored Distillation for Positive Congruent Depth Completion.
https://arxiv.org/abs/2203.16034

@inproceedings{liu2022monitored,
  title={Monitored distillation for positive congruent depth completion},
  author={Liu, Tian Yu and Agrawal, Parth and Chen, Allison and Hong, Byung-Woo and Wong, Alex},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
'''

import os, gdown

PRETRAINED_MODELS_DIRPATH = 'external_models'

GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

# RGB_GUIDANCE_UNCERTAINTY pretrained models
RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_MODELS_DIRPATH = \
    os.path.join(PRETRAINED_MODELS_DIRPATH, 'rgb_guidance_uncertainty')

# RGB_GUIDANCE_UNCERTAINTY pretrained KITTI model
RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_DIRPATH = \
    os.path.join(RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_MODELS_DIRPATH, 'kitti')

RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1jtiYb7tOuidMZKB_S9aq8EP-Bg3EiHr3')

RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_FILENAME = 'model_best_epoch.pth.tar'
RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_FILEPATH = \
    os.path.join(RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_DIRPATH, RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_FILENAME)

# ERFNet pretrained model (for initialization)
ERFNET_PRETRAINED_CITYSCAPES_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1GURV8mxzdpKBWxk5LOWaPP5PZsrLVEdd')

ERFNET_PRETRAINED_CITYSCAPES_MODEL_FILENAME = 'erfnet_pretrained.pth'
ERFNET_PRETRAINED_CITYSCAPES_MODEL_FILEPATH = \
    os.path.join(RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_DIRPATH, ERFNET_PRETRAINED_CITYSCAPES_MODEL_FILENAME)

def setup_rgb_guidance_uncertainty_model():

    # Download pretrained model
    dirpaths = [
        RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_MODELS_DIRPATH,
        RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_DIRPATH
    ]

    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    if not os.path.exists(RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_FILEPATH):
        print("Downloading {} to {}".format(
            RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_FILENAME, RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_FILEPATH))

        gdown.download(RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_URL, RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_FILEPATH, quiet=False)
    else:
        print("Found {} at {}".format(
            RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_FILENAME, RGB_GUIDANCE_UNCERTAINTY_PRETRAINED_KITTI_MODEL_FILEPATH))

    if not os.path.exists(ERFNET_PRETRAINED_CITYSCAPES_MODEL_FILEPATH):
        print("Downloading {} to {}".format(
            ERFNET_PRETRAINED_CITYSCAPES_MODEL_FILENAME, ERFNET_PRETRAINED_CITYSCAPES_MODEL_FILEPATH))

        gdown.download(ERFNET_PRETRAINED_CITYSCAPES_MODEL_URL, ERFNET_PRETRAINED_CITYSCAPES_MODEL_FILEPATH, quiet=False)
    else:
        print("Found {} at {}".format(
            ERFNET_PRETRAINED_CITYSCAPES_MODEL_FILENAME, ERFNET_PRETRAINED_CITYSCAPES_MODEL_FILEPATH))


if __name__ == "__main__":
    setup_rgb_guidance_uncertainty_model()
