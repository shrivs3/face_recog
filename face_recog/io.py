import os
from PIL import Image
import numpy as np


class ImageDataset(object):
    """
    Image dataset object. 

    Directory Structure of image_path:
            train
            ├── ben_afflek
            |   ├── img_1   
            |   └── img_2 
            ├── elton_john
            |   ├── img_1   
            |   └── img_2 
            ├── jerry_seinfeld
            |   ├── img_1   
            |   └── img_2 
            ├── madonna
            |   ├── img_1   
            |   └── img_2 
            └── mindy_kaling
                ├── img_1   
                └── img_2 
    """
    def __init__(self, image_path:str=None):
        self.image_path = image_path
        self.img_array_detected = None

        self.X = []
        self.y = []

        if self.image_path is not None:
            self.check_path_exist(self.image_path)

    def check_path_exist(self, path):
        """
        Checks if the provided path exists in the system
        """
        if os.path.exists(path):
            return 
        else:
            raise Exception('The provided path '+str(path)+' does not exists.')

    def load_data(self, convert_xy=False):
        """
        Load all the data at once.
        """
        if self.image_path is None:
            raise Exception('The path for image directory is not provided.')

        self.img_array_detected = {}
        # iterating over the folders.
        for folder in os.listdir(self.image_path):
            self.img_array_detected[folder] = []
            folder_path = os.path.join(self.image_path, folder)
            for file_name in os.listdir(folder_path):
                # checking to read only jpeg files
                path = os.path.join(folder_path, file_name)                            
                ### NOTE: The image should be 3 dimensional or RGB format
                img = Image.open(path)            
                # converting to numpy array
                img_array =  np.asarray(img)
                self.img_array_detected[folder].append(img_array)

        # if convert to X, y format for ML
        if convert_xy:
            for key in self.img_array_detected:
                self.X.extend(self.img_array_detected[key])
                self.y.extend([key for i in range(len(self.img_array_detected[key]))])

        return self.X, self.y

