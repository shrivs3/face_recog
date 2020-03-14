from sklearn.base import TransformerMixin
from mtcnn import MTCNN
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt

class FaceDetectorTransformer(TransformerMixin):
    def __init__(self, final_size=None):
        self.final_size = final_size
        self.label_mask = []

    def fit(self, X):
        return self

    def transform(self, X):
        bounding_box = MTCNN()

        if self.final_size is None:
            self.final_size = (160, 160)
    
        X_transformed = []
        # detecting face
        for i, img_array in enumerate(X):
            try:
                # finding the bounding box boundaries
                boundaries =  bounding_box.detect_faces(img_array)
                x_1, y_1, w, h = boundaries[0]['box']
                x_2 = x_1 + w
                y_2 = y_1 + h
                
                # cropping image
                focus_area = img_array[y_1:y_2, x_1:x_2]
                img_cropped = Image.fromarray(focus_area, 'RGB')
                
                # reshaping for allowing for model input
                img_cropped = img_cropped.resize(self.final_size)
                img_cropped_arr = np.asarray(img_cropped)
                X_transformed.append(img_cropped_arr)

                self.label_mask.append(True)

            except Exception as e:
                print('Error finding bounding box:' , e)
                self.label_mask.append(False)

        return np.array(X_transformed)

    def apply_label_mask(self, y):
        if self.label_mask is not None:
            return [v for i,v in enumerate(y) if self.label_mask[i]]
