from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model,load_model
import numpy as np


class FeatureExtractor:
    def __init__(self):
        self.base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='max') # change smaller model
        self.base_model.load_weights('static/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
        self.base_model._make_predict_function()

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        feature = self.base_model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize
