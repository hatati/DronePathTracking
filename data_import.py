import os
import cv2
from tqdm import tqdm
import random
import numpy as np
import pickle


DATADIR = "C:/Users/nermi/Desktop/dataset/Scaled Images"

CATEGORIES = ["forward", "left", "right"]

IMG_SIZE = 70

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # create path to forward, left and right
        class_num = CATEGORIES.index(category) # get the classification  (0, 1 or a 2). 0=forward 1=left 2=right

        for img in tqdm(os.listdir(path)): # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print("Number of samples: ", len(training_data))

#Shuffle the data
random.shuffle(training_data)


#Create dataset
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

#X has to be a numpy array while y can stay a list
#-1: take all features, 3: image is a RGB image (3 color channels)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


#Save the training data to a pickle file
pickle_out = open("Dataset/X_RGB.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Dataset/y_RGB.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()












