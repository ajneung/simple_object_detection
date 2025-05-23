#----------begin fix pytorch 2.6 -------------------
import torch

# Save the original torch.load function
_original_torch_load = torch.load

# Define a new function that forces weights_only=False
def custom_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

# Override torch.load globally
torch.load = custom_torch_load
#-------end fix pytorch 2.6------------------------

from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os
from PIL import Image
import logging
# from my_keras_model import KerasModel
import time
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

yolo_model_paths = {
    'yolo.pt': r'disease_best.pt'
}



# Add the safe global to allow loading the DetectionModel class
torch.serialization.add_safe_globals([DetectionModel])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/imgpred', methods=['POST'])
def imgpred():
    model_name = request.form.get('model')
    if not model_name:
        logging.error("Model name is missing.")
        return redirect(url_for('home'))
    image_file = request.files.get('image')
    if not image_file:
        logging.error("Image file is missing.")
        return redirect(url_for('home'))
    original_filename = secure_filename(image_file.filename)
    new_filename = f"{int(time.time())}_{original_filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    image_file.save(image_path)
    logging.info(f"Image saved at {image_path}")
    try:
        class_name = None
        percentage = 0
        characteristics = {}
        result_image_filename = new_filename
        start_time = time.time()
        if model_name in yolo_model_paths:
            model = YOLO(yolo_model_paths[model_name])
            logging.info(f"YOLO Model {model_name} loaded successfully.")
            class_name, percentage, characteristics, result_image_filename = predict_yolo(image_path, model)
        else:
            logging.error(f"Invalid model name selected: {model_name}")
            return redirect(url_for('home'))
        end_time = time.time()
        inference_time = round(end_time - start_time, 2)

        return render_template('index.html',
                               image_path=result_image_filename,
                               class_name=class_name,
                               percentage=percentage,
                               characteristics=characteristics,
                               inference_time=inference_time,
                               model_name=model_name)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return redirect(url_for('home'))

def predict_yolo(image_path, model):
    results = model(image_path)
    result_image_filename = f"result_{os.path.basename(image_path)}"
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_image_filename)
    annotated_frame = results[0].plot()
    im_rgb = Image.fromarray(annotated_frame[..., ::-1])
    im_rgb.save(result_image_path)
    logging.info(f"Result image saved at {result_image_path}")
    max_confidence = 0
    max_class = None
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)]
        confidence = float(box.conf) * 100
        if confidence > max_confidence:
            max_confidence = confidence
            max_class = cls_name
    detected_characteristics = max_class
    if max_class is None:
        max_class = "None"
    return max_class, round(max_confidence, 2), detected_characteristics, result_image_filename


if __name__ == "__main__":
    app.run(debug=True)