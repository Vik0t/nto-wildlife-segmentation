from ultralytics import YOLO
import numpy as np
import cv2 
import random
class DetectionModel():
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path)

    def add_noise(self, img):
        equ = cv2.equalizeHist(img)
        res = np.hstack((img,equ))
        return res
    
    def denoise(self, img):
        return cv2.fastNlMeansDenoising(img,  None, 30, 7, 21)
    
    def predict(self,image_path):
        return self.model(image_path)[0]
    
    def get_conf(self, prediction):
        conf = 0
        els = 0
        for el in prediction:
            conf += el.conf
            els += 1
        return conf/els 

    def ensemble_predict(self,image_path):
        noisy_pred = self.predict(self.add_noise(cv2.imread(image_path)))
        denoised_pred = self.predict(self.denoise(cv2.imread(image_path)))
        normal_pred = self.predict(image_path)
        
        noisy_conf = self.get_conf(noisy_pred)
        denoised_conf = self.get_conf(denoised_pred)
        normal_conf = self.get_conf(normal_pred)
        
        preds = {noisy_conf: noisy_pred, denoised_conf: denoised_pred, normal_conf: normal_pred}
        return preds[max(normal_conf, denoised_conf, noisy_conf)]

