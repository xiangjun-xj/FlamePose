import face_alignment
from skimage import io
import cv2
import numpy as np
import os
import os.path as osp
import argparse



""" fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

preds = fa.get_landmarks_from_directory('./data/TestFace_origin/images/')
print(len(preds)) """

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,
                    default='2-346', help='scene name')

args = parser.parse_args()


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#from single image
""" input = io.imread('./data/TestFace_origin/images/007.jpg')
print(type(input))
print(input.shape)
preds = fa.get_landmarks(input)
a=preds[0]
print(type(a))
print(a.shape)
for i in range(68):
    cv2.circle(input, (int(a[i,0]),int(a[i,1])), 2, (0,255,0), 3 , cv2.LINE_AA)
path = "./data_landmark/1.jpg"
cv2.imwrite(path, input) """

#from image directory
image_base_dir = './data'
lmk_base_dir = './data_landmark'
imgpath = os.path.join(image_base_dir, args.id)
image_path = os.path.join(imgpath, 'images_pad')
save_path = os.path.join(lmk_base_dir, args.id)
if not os.path.exists(save_path):
        os.makedirs(save_path)
landmarks_path = os.path.join(save_path, 'landmarks.npy')
preds = fa.get_landmarks_from_directory(image_path)
landmarks = np.array([])
length = 0
""" for key, value in sorted(preds.items()):
    print(key[-7:-4]) """
for key, value in sorted(preds.items()):
    value = np.array(value)
    value = np.array([value[0]])#[1,68,2]
    if length==0:
        landmarks = value
    else:
        landmarks = np.concatenate((landmarks, value), axis=0) 
    single_path = os.path.join(save_path, '{:03d}.npy'.format(length))
    np.save(single_path, value)
    length = length+1
    
np.save(landmarks_path, landmarks)


