'''

Goals:          Copyb files from one folder to another

input::         Set paths in main

output::        files will be copied from source to destination folder


Mohammad Zohaib
PAVIS | IIT, Genova, Italy
mohammad.zohaib@iit.it
engr.mz@hotmail.com
2020/11/15

'''

# My updates
import sys, os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import glob
import os
import copy
from tqdm import tqdm


def main():

    input_dir = '/media/mz/mz/Datasets/onet_shapenet/ShapeNet_3cat/04379243'
    output_dir = '/media/mz/mz/Datasets/onet_shapenet/realistic_dataset/04379243'

    folder_list = sorted(glob.glob('{}/*/'.format(input_dir)))
    for dir_ in tqdm(folder_list):
        model_path = os.path.join(dir_, 'model.binvox')
        pointcloud_path = os.path.join(dir_, 'pointcloud.npz')
        points_path = os.path.join(dir_, 'points.npz')

        model_out_path = os.path.join(output_dir, dir_.split('/')[-2], 'model.binvox')
        pointcloud_out_path = os.path.join(output_dir, dir_.split('/')[-2], 'pointcloud.npz')
        points_out_path = os.path.join(output_dir, dir_.split('/')[-2], 'points.npz')

        shutil.copy(model_path, model_out_path)
        shutil.copy(pointcloud_path, pointcloud_out_path)
        shutil.copy(points_path, points_out_path)


if __name__ == "__main__":
    main()



