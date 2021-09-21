'''

Goals:          Building an data loading pipeline (images, labels from npz files)

Description:    The script loads images from "images" folder
                lables (points, occupancies from npz files)

input::         Images from folder "img_choy2016"
                dataset_dir = 'dataset/mini_shapenet/ShapeNet'


output::        Loads images and corresponding GT points simultaneously


Mohammad Zohaib
PAVIS | IIT, Genova, Italy
mohammad.zohaib@iit.it
engr.mz@hotmail.com
2020/11/15

'''

# My updates
import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import yaml
import random
import csv

#to add
import cv2
import glob
import os
import copy
from tqdm import tqdm

class AddBackground:
    def __init__(self, imgs_path, bg_path):
        self.image_h = 224
        self.image_w = 224
        self.image_d = 3
        self.bg_dir = bg_path
        self.images_dir = imgs_path
        self.metadata_file = 'metadata.yaml'
        self.image_folder = 'img_choy2016'
        self.output_folder = 'output_background_images_table'
        self.class_dir = self.get_imgs_dir()    # not using it
        self.images_list = self.get_images_list()        # list of (path,images)
        self.back_grounds = self.load_backgrounds()




    ''' Read "metadata.yaml" file and generates list containing paths of all the classes '''
    def get_imgs_dir(self):
        with open(os.path.join(self.images_dir, self.metadata_file)) as file:
            yaml_file = yaml.load(file)
        k = [x for x in yaml_file.keys()]
        return [os.path.join(self.images_dir, x) for x in k]

    '''Return list of all the images and point labels'''
    def get_images_list(self):
        print("Please wait - Creating images list")
        imageslist = []
        dataset = sorted(glob.glob('{}/04379243/*/'.format(self.images_dir)))
        for dir_ in tqdm(dataset):
            imgs_path = os.path.join(dir_, self.image_folder)
            imgs_path = [x for x in os.listdir(imgs_path) if not x.endswith('.npz')]
            imgs_path.sort()
            imgs_path = [os.path.join(dir_, self.image_folder, x) for x in imgs_path]

            for path in imgs_path:
                imageslist.append((path))
                # imageslist.append((path, self.read_image(path)))

        print("Loaded {} images".format(len(imageslist)))
        return imageslist

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_w, self.image_h), interpolation=cv2.INTER_CUBIC)
        return image

    def load_backgrounds(self):
        # load random background
        print("Please wait - loading background")
        back_grounds = []
        bk_lst = glob.glob(self.bg_dir + '/*/*.jpg')
        for bk_path in tqdm(bk_lst):
            back_ground = cv2.imread(bk_path)
            back_grounds.append(cv2.resize(back_ground, (self.image_w, self.image_h), interpolation=cv2.INTER_CUBIC))

        print("Loaded {} backgrounds".format(len(back_grounds)))
        return back_grounds

    def save_background_img(self):
        print("Please wait - Updating background")
        output_dir =''
        for img_path in tqdm(self.images_list):
            img = self.read_image(img_path)
            updated_img = copy.deepcopy(img)
            # img_path = imgs[0]
            updated_path = img_path.split('/')
            updated_path[-5] = self.output_folder
            output_dir = os.path.join('/', *updated_path[0:-1])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            updated_path = os.path.join('/', *updated_path)
            bk_index = random.randint(0, len(self.back_grounds) - 1)
            bk_img = copy.deepcopy(self.back_grounds[bk_index])

            # Separate mask and object
            bg_mask = np.where(updated_img >= 245, 1.0, 0.0).astype(np.uint8)  # BG is 1, obj = 0
            img_mask = np.where(updated_img > 245, 0.0, 1.0).astype(np.uint8)  # BG is 0, obj = 1
            bk_img *= bg_mask
            updated_img *= img_mask

            # combine separated mask and object
            updated_img += bk_img

            # cv2.imwrite('bg_mask.jpg', bg_mask)
            # cv2.imwrite('img_mask.jpg', img_mask)
            # cv2.imwrite('updated_img.jpg', updated_img)

            cv2.imwrite(updated_path, updated_img)
        print("Backgound images are saved at : {}".format(output_dir))
        # return updated_img



def main():

    images_dir = '/sample_dataset/synthetic_images'
    background_dir = '/sample_dataset/backgrounds'

#    images_dir = '/home/mz/code/onet_initial/data/ShapeNet'
#    background_dir = '/home/mz/code/onet_initial/data/back_grounds'
    dataset = AddBackground(images_dir, background_dir)
    dataset.save_background_img()


if __name__ == "__main__":
    main()















#
#
#
#
#
# ## ------------- Comparing encoder losses ------------ ##
# from torch.nn import functional as F
# import matplotlib.pyplot as plt
# import glob
# import random
# import cv2
# import copy
# import numpy as np
#
#
#
# def show_tensor_image(img, name='img'):
#     '''
#     Args:
#         img: image in tensor format [3,224,224]
#
#     Returns:    Reshape and display the image
#     '''
#     plt.imshow(img.cpu().numpy()[0], cmap='gray')
#     plt.savefig('{}.jpg'.format(name))
#     # plt.show()
#
#
#
#
# def load_imgs(path):
#     p = batch.get('points').to(device)
#     occ = batch.get('points.occ').to(device)
#     white_inputs = batch.get('inputs', torch.empty(p.size(0), 0)).to(device)
#     color_inputs = add_background(white_inputs)#.to(device)
#
#     white_logits = model.encoder(white_inputs)
#     color_logits = model.encoder(color_inputs)
#     loss = F.l1_loss(white_logits,color_logits)
#     # print(loss)
#     return loss, white_inputs[0], color_inputs[0]
#
#
# def load_images(path):
#     # load random background
#     img_lst = []
#     bk_lst = glob.glob(path + '/*/*.jpg')
#     for bk_path in bk_lst:
#         back_ground = cv2.imread(bk_path)
#         back_ground = cv2.resize(back_ground, [224,224])
#         back_ground = cv2.normalize(back_ground, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#
#
#     return back_grounds
#
#
#
# def main():
#     synthetic_dir = ''
#     background_dir = ''
#     synthetic_img_lst=load_images(synthetic_dir)
#     synthetic_img_lst = load_images(background_dir)
#
# if __name__ == "__main__":
#     main()
#
