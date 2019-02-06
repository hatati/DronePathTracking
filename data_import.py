import tensorflow as tf
import os
import csv

data_root = "C:/Users/nermi/Desktop/dataset/test/"
labels_root = "C:/Users/nermi/Desktop/dataset/labels.csv"
dataset = []


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [60, 60])
    # image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def load_labels(path=labels_root):
    labels = []
    with open(path, 'r') as csvfile:
        temp = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in temp:
            labels.append(row)

    return labels


def load_images(path=data_root):
    images = []
    for root, dirs, files in os.walk(data_root):
        for filename in files:
            image = load_and_preprocess_image(data_root + filename)
            images.append(image)

    return images

#load_labels()
#print(labels)
load_images()