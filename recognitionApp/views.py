import inline as inline
import matplotlib
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from recognitionApp.models import sentImagetoDatabase
from recognitionApp.wavelet import w2d
from recognitionApp.croppedImage import get_cropped_image_if_2_eyes
from recognitionApp.scaling import scaling
from recognitionApp.celeb_names import celeb_names
import pickle
import json
import sklearn
from sklearn.preprocessing import StandardScaler
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# Create your views here.

def home(request):
    name = 'Rohith'
    return HttpResponse('Hi this is {}'.format(name))

def imageUpload(request):
    return render(request, 'upload.html')

def read_model():
    file_model = open(r'Celebrity Face Recognition/image_model.sav', 'rb')
    model = pickle.load(file_model)
    return model

def predict_celebrity(img_path):
    name_class = {}
    roi_color = get_cropped_image_if_2_eyes(img_path)
    if len(roi_color) != 0:
        for i in range(len(roi_color)):
            cropped_image = np.array(roi_color[i])
            im_har = w2d(cropped_image, 'db1', 5)
            scaled_wavelet = cv2.resize(im_har, (32, 32))
            scaled_original = cv2.resize(roi_color[i], (32, 32))
            combined = np.vstack((scaled_original.reshape(32*32*3, 1), scaled_wavelet.reshape(32*32, 1)))
            test = np.array(combined).reshape(1, len(combined)).astype(float)
            test_scaled = scaling(test.reshape(-1, 1))
            for name, class_ in celeb_names().items():
                if class_ == read_model().predict(test_scaled.reshape(1, -1))[0]:
                    probability = read_model().predict_proba(test_scaled.reshape(1, -1))[0][class_]
                    name_class[name] = probability
        return name_class
    return None

def getImage(request):
    if request.method == 'POST':
        imgfile = request.FILES.get('imgfile')
        k = sentImagetoDatabase(img=imgfile)
        k.save()

        import os
        for im in os.scandir('./media/images'):
            res = predict_celebrity(im.path)
            os.remove(path=im.path)

        if res:
            ans = {}
            for k, v in res.items():
                k = k.replace('_', " ")
                ans[k.title()] = round(v, 3)
            return render(request, 'result.html', {'res': ans})
        else:
            return render(request, 'result.html', None)
