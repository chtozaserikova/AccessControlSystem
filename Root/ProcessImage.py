from PIL import Image
import numpy as np 
import torch

class ProcessImage:
    def __init__(self):
        pass
    
    @staticmethod
    def Process(image_array): # Takes in a Numpy array
        pil_image = np.array(Image.fromarray(image_array).resize((100,100))) 
        img = np.moveaxis(pil_image , -1 ,0)[1,:,:]
        img = np.expand_dims(img , 0)
        img = np.expand_dims(img , 0)

        return img