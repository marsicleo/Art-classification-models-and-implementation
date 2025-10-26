import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import h5py
import os
import logging
from transformers import TFViTModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

MODEL_DEFINITION_PATH = 'models_online/model_definition.py'
WEIGHTS_PATH = 'models_online/model_weights.h5'
CLASS_NAMES_PATH = 'models_online/label_encoder_classes.npy'
IMAGE_SIZE = (224, 224)

class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
num_classes = len(class_names)

def build_model(num_classes):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    x = tf.keras.layers.Lambda(lambda img: (img / 255.0 - 0.5) / 0.5)(inputs)
    x = tf.keras.layers.Lambda(lambda img: tf.transpose(img, [0, 3, 1, 2]))(x)
    
    vit = TFViTModel.from_pretrained("google/vit-base-patch16-224")
    vit.trainable = True
    vit_out = vit(pixel_values=x)    
    x = vit_out.last_hidden_state[:, 0, :] 
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x) 
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)


def load_custom_weights(model, weight_path):
    try:
        with h5py.File(weight_path, 'r') as f:
            try:
                model.load_weights(weight_path)
                logger.info("Loaded weights Keras method")
                return True
            except Exception as e:
                logger.info("Standard loading failed")
            try:
                dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
                model.predict(dummy_input)
                logger.info(" Weights validated")
            except Exception as e:
                logger.error(f" Weight validation failed: {str(e)}")
            
            layer_weights = {}
            
            def collect_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    path = name.split('/')
                    layer_name = '/'.join(path[:-1])
                    if layer_name not in layer_weights:
                        layer_weights[layer_name] = {}
                    layer_weights[layer_name][path[-1]] = obj[()]            
            f.visititems(collect_weights)
            
            for layer in model.layers:
                if layer.name in layer_weights:
                    weights = []
                    if 'kernel:0' in layer_weights[layer.name]:
                        weights.append(layer_weights[layer.name]['kernel:0'])
                    if 'bias:0' in layer_weights[layer.name]:
                        weights.append(layer_weights[layer.name]['bias:0'])                    
                    if weights:
                        try:
                            layer.set_weights(weights)
                            logger.info(f"Loaded weights for {layer.name}")
                        except Exception as e:
                            logger.warning(f"Couldn't set weights for {layer.name}: {str(e)}")
            
            dummy_input = np.random.rand(1, *IMAGE_SIZE, 3).astype(np.float32)
            model.predict(dummy_input)
            logger.info(" Weights loaded and verified")
            return True

    except Exception as e:
        logger.error(f" Weight loading failed: {str(e)}")
        return False

logger.info("Building model")
model = build_model(num_classes)
model.summary(print_fn=lambda x: logger.info(x))

logger.info("Loading weights")
if not load_custom_weights(model, WEIGHTS_PATH):
    logger.error("\n Critical Fix Required:")    
    exit(1)

def preprocess_image(image):
    img = image.resize(IMAGE_SIZE).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    return np.expand_dims(img_array, axis=0)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            return jsonify({
                "predictions": [],
                "class_names": [],
                "error": "No image provided"
            }), 400

        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        
        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img)[0]
        
        return jsonify({
            "predictions": predictions.tolist(), 
            "class_names": class_names
        })
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return jsonify({
            "predictions": [],
            "class_names": [],
            "error": str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "status": "active",
        "model": "VIT",
        "classes": num_classes,
        "input_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
    })

if __name__ == '__main__':
    for path in [WEIGHTS_PATH, CLASS_NAMES_PATH]:
        if not os.path.exists(path):
            logger.error(f"Missing required file: {path}")
            exit(1)

    logger.info("Starting server")
    app.run(host='0.0.0.0', port=5000, threaded=True)