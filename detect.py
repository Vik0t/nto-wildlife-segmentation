from ultralytics import YOLO
import numpy as np
import cv2 
import random
class DetectionModel():
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path)

    def equalize_hist(self, img):
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        # convert back to RGB color-space from YCrCb
        equ = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        #res = np.hstack((img,equ))
        return equ #res
    
    def denoise(self, img):
        return cv2.fastNlMeansDenoising(img,  None, 30, 7, 21)
    
    def predict(self,image_path):
        return self.model(image_path)[0]
    
    def get_conf(self, prediction):
        conf = 0
        els = 0
        for el in prediction:
            conf += el.boxes.conf
            els += 1
        return conf/els 

    def ensemble_predict(self,image_path):
        equ_pred = self.predict(self.equalize_hist(cv2.imread(image_path)))
        denoised_pred = self.predict(self.denoise(cv2.imread(image_path)))
        normal_pred = self.predict(cv2.imread(image_path))
        
        equ_conf = self.get_conf(equ_pred)
        denoised_conf = self.get_conf(denoised_pred)
        normal_conf = self.get_conf(normal_pred)
        
        preds = {equ_conf: equ_pred, denoised_conf: denoised_pred, normal_conf: normal_pred}
        print("equ             denoised           normal")
        print(equ_conf.cpu().numpy(), denoised_conf.cpu().numpy(), normal_conf.cpu().numpy())
        return preds[max(normal_conf, denoised_conf, equ_conf)]

