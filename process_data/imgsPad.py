import numpy as np
import cv2
import os
import sys
import imageio
import argparse



def pad_images(img, target_size):
    w, h, _ = img.shape

    if h < w:
        pad_img = np.zeros([w, w, 3])
        center = w / 2.0
        h1 = int(center - h / 2.0)
        h2 = h1 + h
        pad_img[:w, h1:h2, :] = img[:, :, :]
        res_img = cv2.resize(pad_img, (target_size, target_size))
    else:
        pad_img = np.zeros([h, h, 3])
        center = h / 2.0
        w1 = int(center - w / 2.0)
        w2 = w1 + w
        pad_img[w1:w2, :h, :] = img[:, :, :]
        res_img = cv2.resize(pad_img, (target_size, target_size))

    return res_img

def pad_imgdir(scenedir):
    imgdir = os.path.join(scenedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgs = np.stack([imageio.imread(img)/255. for img in imgs], 0)
    img_pad_dir = os.path.join(scenedir, 'images_pad')
    if not os.path.exists(img_pad_dir):
        os.makedirs(img_pad_dir)
    maxsize = imgs.shape[1] if imgs.shape[1]>imgs.shape[2] else imgs.shape[2]
    for index in range(imgs.shape[0]):
        pad_img = pad_images(imgs[index], maxsize)
        imageio.imwrite(os.path.join(img_pad_dir, 'image{:03d}.png'.format(index)), (255*pad_img).astype(np.uint8))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str,
                        default='2-346', help='scene name')

    args = parser.parse_args()
    basedir = './data'
    scenedir = os.path.join(basedir, args.id)
    pad_imgdir(scenedir)