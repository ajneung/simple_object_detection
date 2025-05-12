from keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
       
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

class KerasModel:
    def __init__(self, model_path, labels_path):
        # Load Keras model
        self.model = self.load_model(model_path)
        # Load class names from labels.txt
        self.class_names = self.load_class_names(labels_path)
        
    def load_model(self, model_path):
        # Ensure the model file has a .h5 extension
        if not model_path.lower().endswith('.h5'):
            raise ValueError("Only .h5 files are supported for loading models.")
        return load_model(model_path, compile=False, custom_objects={
            'DepthwiseConv2D': CustomDepthwiseConv2D
        })
    
    def load_class_names(self, labels_path):
       
        with open(labels_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    
    def preprocess_image(self, image_path):
        
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        return data

    def predict(self, image_path):
       
        data = self.preprocess_image(image_path)
        prediction = self.model.predict(data)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]
        return class_name, confidence_score
