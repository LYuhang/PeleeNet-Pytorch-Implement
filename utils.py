# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageEnhance
from tqdm import tqdm
from config import Config
import pickle

'''
This part is used to preprocess image before using them.
You should run this file at the beginning.
'''


def gen_image_path_file(path, label_dict=None):
    '''
    This function is used to generate image path file.
    It is a txt file, format:
        [image_path] [name]
        (the name of the picture is the class name)
    The path is the root path of the data, the txt file
    will be saved in root path.If label_dict is given,
    the format will be:
        [image_path] [label]
    :param path: the image path
    :return: None
    '''
    if not os.path.exists(path):
        raise Exception("==> The path does not exists : %s" % path)

    names = []  # store the class name of the image
    im_paths = []  # store the im_path of the image
    print("==> Begin to generate names and paths")
    for root, dirs, files in tqdm(os.walk(path, topdown=False)):
        name = os.path.split(root)[1]
        name = name.split("_")[0]      # get the class name from the dirname
        for f in files:
            if os.path.splitext(f)[1] == ".jpg":
                names.append(name)
                im_paths.append(os.path.join(root, f))

    print("==> Generate image path txt file")
    # Gen txt file
    if not label_dict:
        lines = ["%s %s" % (ph, nm) for ph, nm in zip(im_paths, names)]  # gen "[path] [name]" line
        print("==> Generate image_name file")
        txt_path = os.path.join(path, "image_name.txt")
        with open(txt_path, "w+") as fp:
            fp.write('\n'.join(lines))
    else:
        # transform the image name to label
        labels = list(map(lambda x: str(label_dict[x]), names)) # transform from image name to label
        lines = ["%s %s" % (ph, lb) for ph, lb in zip(im_paths, labels)]
        txt_path = os.path.join(path, "image_label.txt")
        with open(txt_path, "w+") as fp:
            fp.write("\n".join(lines))


def data_augment(image_path):
    '''
    This function used to apply image augmentation, including
    => light_brighten
    => light_darken
    => color_enhance
    => color_darken
    => contrast_enhance
    => contrast_weaken
    => sharpness_enhance
    => sharpness_weaken
    This function reads image with PIL.Image method and augment,
    after that, store the augmented image with "_[augment_method].jpg"
    suffix at the same path
    :param image_path : images path
    :return: None
    '''
    im_names = os.listdir(image_path)
    im_paths = [os.path.join(image_path, i) for i in im_names]
    for name, path in zip(im_names, im_paths):
        name = os.path.splitext(name)[0]
        image = Image.open(path) # open the image with PIL.Image method

        # light brighten and darken
        light_bri = ImageEnhance.Brightness(image)
        light_bri_im = light_bri.enhance(1.5)
        light_bri_im.save(os.path.join(image_path, "%s_light_brighten.jpg" % name))
        light_dar_im = light_bri.enhance(0.8)
        light_dar_im.save(os.path.join(image_path, "%s_light_darken.jpg" % name))
        # color enhance and darken
        color_enh = ImageEnhance.Color(image)
        color_enh_im = color_enh.enhance(1.5)
        color_enh_im.save(os.path.join(image_path, "%s_color_enhance.jpg" % name))
        color_dar_im = color_enh.enhance(0.8)
        color_dar_im.save(os.path.join(image_path, "%s_color_darken.jpg" % name))
        # contrast enhance and weaken
        contrast_enh = ImageEnhance.Contrast(image)
        contrast_enh_im = contrast_enh.enhance(1.5)
        contrast_enh_im.save(os.path.join(image_path, "%s_contrast_enhance.jpg" % name))
        contrast_dar_im = contrast_enh.enhance(0.8)
        contrast_dar_im.save(os.path.join(image_path, "%s_contrast_darken.jpg" % name))
        # sharpness enhance and weaken
        sharpness_enh = ImageEnhance.Sharpness(image)
        sharpness_enh_im = sharpness_enh.enhance(3.0)
        sharpness_enh_im.save(os.path.join(image_path, "%s_sharpness_enhance.jpg" % name))
        sharpness_dar_im = sharpness_enh.enhance(0.8)
        sharpness_dar_im.save(os.path.join(image_path, "%s_sharpness_darken.jpg" % name))


def all_data_augment(path):
    '''
    This function is used to augment all images
    :param path: the root path of the training images
    :return: None
    '''
    dirnames = os.listdir(path)
    dirpaths = [os.path.join(path, i) for i in dirnames] # get all the dirnames
    dirpaths = list(filter(lambda x: os.path.isdir(x), dirpaths))
    print("==> Apply image augmentation.")
    for dp in tqdm(dirpaths):
        data_augment(dp)

if __name__ == "__main__":
    # Initialize the config
    conf = Config()

    # generate image path file
    #gen_image_path_file(conf.RAW_TRAIN_DATA)

    # generate image path and label file
    gen_image_path_file(conf.RAW_TRAIN_DATA, pickle.load(open(conf.NAME_TO_LABEL_PATH, "rb")))

    # image augmentation
    #all_data_augment(conf.RAW_TRAIN_DATA)