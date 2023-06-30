import cv2
from keras.models import load_model
import numpy as np
import math

from utils.datasets import get_labels
from utils.inference import load_detection_model


import datetime

def loadModel(detection_model_path,emotion_model_path):
    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    return face_detection,emotion_classifier

def getLabels(label='fer2013'):
    emotion_labels = get_labels(label)
    return emotion_labels

def getEmotionText(emotion_labels,emotion_prediction):
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    return emotion_text

def getEmotionProbability(emotion_prediction):
    return np.max(emotion_prediction)

