import os
import numpy as np
import pandas as pd
from cv2 import cv2
from PIL import Image
import glob
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import LabelEncoder

#load the data Folder Path
datasets = ImageFolder('../image/')
print("This is the categories:")
print(datasets.class_to_idx)

#load the data and resize
image_array=[]
label = []
for img in datasets.imgs:
    image= cv2.imread(img[0])
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    image_array.append(np.array(size_image))
    label.append(img[1])
    
print("This is the length of the entirely dataset:")
print(len(image_array), len(label))
print("This is the size of the entirely dataset:")
print((np.array(image_array)).shape)

la=LabelEncoder()
labels=la.fit_transform(label)

images = np.array(image_array)
np.save("image",images)
np.save("labels",labels)
