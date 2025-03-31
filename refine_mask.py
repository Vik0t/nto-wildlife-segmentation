import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine

def refine_mask(img, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # model_path can also be specified here
    # This step takes some time to load the model
    refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

    # Fast - Global step only.
    # Smaller L -> Less memory usage; faster in fast mode.
    output = refiner.refine(img, mask, fast=False, L=900) 

    # this line to save output
    #cv2.imwrite('output.png', output)

    return output