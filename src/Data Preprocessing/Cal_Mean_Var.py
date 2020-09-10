import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

#root2 = ""
#root1 = ""
R_channel = 0
G_channel = 0
B_channel = 0
img_num = 0
for file in os.listdir(root1):
    file_path = os.path.join(root1,file)
    for image in os.listdir(file_path):
        image_path = os.path.join(file_path,image)
        img = imread(image_path) / 255.0
        R_channel = R_channel + np.sum(img[:,:,2])
        G_channel = G_channel + np.sum(img[:,:,1])
        B_channel = B_channel + np.sum(img[:,:,0])
        #print(B_channel)
        img_num += 1
#print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
"""
for file in os.listdir(root2):
    file_path = os.path.join(root2,file)
    for image in os.listdir(file_path):
        image_path = os.path.join(file_path,image)
        img = imread(image_path) / 255.0
        R_channel = R_channel + np.sum(img[:,:,2])
        G_channel = G_channel + np.sum(img[:,:,1])
        B_channel = B_channel + np.sum(img[:,:,0])
        img_num += 1
"""
num = img_num * 2304 * 2304
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num
#print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
R_channel = 0
G_channel = 0
B_channel = 0
img_num = 0
for file in os.listdir(root1):
    file_path = os.path.join(root1,file)
    for image in os.listdir(file_path):
        image_path = os.path.join(file_path,image)
        img = imread(image_path) / 255.0
        R_channel = R_channel + np.sum((img[:,:,2] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:,:,1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:,:,0] - B_mean) ** 2)
"""
for file in os.listdir(root2):
    file_path = os.path.join(root2,file)
    for image in os.listdir(file_path):
        image_path = os.path.join(file_path,image)
        img = imread(image_path) / 255.0
        R_channel = R_channel + np.sum((img[:,:,2] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:,:,1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:,:,0] - B_mean) ** 2)
"""
R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))