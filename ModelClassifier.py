from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from imutils import paths
import numpy as np
import cv2
import os
import sklearn.externals
import joblib


def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features


def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        # [healthy, healthy, parkinson, ....]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features = quantify_image(image)
        data.append(features)
        labels.append(label)

    return (np.array(data), np.array(labels))


trainingPath = "parkinson-dataset/spiral/training"
testingPath = "parkinson-dataset/spiral/testing"
# loading the training and testing data
print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)
# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

trials = {}

model = RandomForestClassifier(n_estimators=100)
forest = model.fit(trainX, trainY)

for i in range(0, 5):
    print("[INFO] training model {} of {}...".format(i + 1,5))
    RandomForestClassifier(n_estimators=100)
    forest = model.fit(trainX, trainY)

joblib_file = "joblib_model.pkl"
joblib.dump(forest, joblib_file)
