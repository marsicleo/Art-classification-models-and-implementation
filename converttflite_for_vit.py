import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from transformers import TFViTModel
import mlflow
import logging


OUTPUT_DIR = "D:/venv sa python 3.9/tate_images"
IMAGE_SIZE = (224, 224)
TFLITE_MODEL_PATH = r"D:\venv sa python 3.9\Diplomski\tflite_folder\vit_model.tflite"
TFLITE_QUANT_MODEL_PATH = r"D:\venv sa python 3.9\Diplomski\tflite_folder\vit_model_quant.tflite"
MLFLOW_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Tate_Collection1")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Model Reconstruction
def build_exact_training_model(num_classes, unfreeze_base=True):
    inputs = Input(shape=IMAGE_SIZE + (3,), dtype=tf.float32)

    x = tf.cast(inputs, tf.float32)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    x = x / 255.0
    x = (x - 0.5) / 0.5
    x = tf.transpose(x, [0, 3, 1, 2])

    vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224")
    vit_model.trainable = unfreeze_base
    vit_outputs = vit_model(pixel_values=x)
    x = vit_outputs.last_hidden_state[:, 0, :]
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def add_preprocessing(model):
    raw_input = Input(shape=(224, 224, 3), dtype=tf.uint8, name="raw_input")
    x = tf.cast(raw_input, tf.float32)
    x = tf.image.resize(x, IMAGE_SIZE)
    outputs = model(x)
    return Model(raw_input, outputs)

# TFLite Conversion
def convert_to_tflite():
    with mlflow.start_run(run_name="ViT_TFLite_Conversion"):
        try:
            classes_path = os.path.join(OUTPUT_DIR, "label_encoder_classes.npy")
            class_names = np.load(classes_path, allow_pickle=True)
            num_classes = len(class_names)
            logger.info(f"Building ViT model for {num_classes} classes")
            model = build_exact_training_model(num_classes)

            weights_path = os.path.join(OUTPUT_DIR, "model_weights.h5")
            logger.info(f"Loading weights from {weights_path}")
            model.load_weights(weights_path)
            logger.info("Weights loaded successfully")
            inference_model = add_preprocessing(model)

            logger.info("Converting to float32 TFLite")
            converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
            tflite_model = converter.convert()
            with open(TFLITE_MODEL_PATH, "wb") as f:
                f.write(tflite_model)
            logger.info(f"Saved float32 model at {TFLITE_MODEL_PATH}")
            def representative_dataset():
                test_dir = OUTPUT_DIR
                image_files = [f for f in os.listdir(test_dir)
                               if f.lower().endswith(('.png','.jpg','.jpeg'))]
                for i, img_file in enumerate(image_files[:100]):
                    img_path = os.path.join(test_dir, img_file)
                    img = tf.io.read_file(img_path)
                    img = tf.io.decode_image(img, channels=3, expand_animations=False)
                    img = tf.image.resize(img, IMAGE_SIZE)
                    img = tf.cast(img, tf.uint8)
                    yield [np.expand_dims(img.numpy(), axis=0)]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            quant_model = converter.convert()

            with open(TFLITE_QUANT_MODEL_PATH, "wb") as f:
                f.write(quant_model)
            logger.info(f"Saved quantized model at {TFLITE_QUANT_MODEL_PATH}")

            mlflow.log_artifact(TFLITE_MODEL_PATH)
            mlflow.log_artifact(TFLITE_QUANT_MODEL_PATH)
            mlflow.log_artifact(classes_path)
            logger.info("ViT TFLite conversion complete")

        except Exception as e:
            logger.error(f"ViT conversion failed: {str(e)}")
            mlflow.log_param("conversion_error", str(e))
            raise

if __name__ == "__main__":
    convert_to_tflite()
