#!/bin/bash

python setup/vkitti/setup_dataset_vkitti_kitti.py \
--conditions \
    clone \
    15-deg-left \
    15-deg-right \
    30-deg-left \
    30-deg-right \
--n_sample_per_image 1 \
