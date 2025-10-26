
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import urllib.request
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from transformers import TFViTModel 
import mlflow
import mlflow.tensorflow
import json
import shutil
import logging
from sklearn.utils import class_weight
import gc
from tensorflow.keras.experimental import CosineDecay

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


DATA_PATH = 'D:/venv sa python 3.9/Diplomski/clean_output.csv'
#DATA_PATH = r'D:\venv sa python 3.9\Diplomski\clean_output - kratki testni.csv'
#OUTPUT_DIR = r"D:\venv sa python 3.9\Diplomski\testno"
OUTPUT_DIR = "D:/venv sa python 3.9/tate_images"
RUN_OUTPUT_DIR = "D:/venv sa python 3.9/Diplomski/runovi"
MLFLOW_URI = "http://127.0.0.1:5000"
RUN_NAME = "Tate_Collection_ViT_B16_V4 "
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8 
EPOCHS_PHASE1 = 5 
EPOCHS_PHASE2 = 20 
MEMORY_SAVE_MODE = True

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Tate_Collection1")

CATEGORY_MAPPING = {
        "Acrylic paint on canvas": "Watercolour",
        "Aquatint on paper": "Watercolour",
        "Bronze": "Sculpture",
        "Plaster": "Sculpture",        
        "Drypoint on paper": "Engraving",
        "Engraving and etching on paper": "Etching",
        "Engraving on paper": "Engraving",               
        "Intaglio print on paper": "Engraving",
        "Line engraving on paper": "Engraving",
        "Linocut on paper": "Engraving",
        "Lithograph on paper": "Engraving",
        "Mezzotint on paper": "Engraving",        
        "Pastel on paper": "Engraving",
        "Pen and ink and graphite on paper": "Graphite",
        "Pen and ink on paper": "ink",
        "Print on paper": "Print",
        "Screenprint and lithograph on paper": "Print",
        "Screenprint on paper": "Print",
        "Graphite": "Graphite",
        "Graphite and chalk on paper": "Graphite",
        "Graphite and gouache on paper": "Graphite",
        "Graphite and ink on paper": "Graphite",
        "Graphite and watercolour on paper": "Graphite",
        "Graphite on paper": "Graphite",
        "Etching": "Etching",
        "Etching and aquatint on paper": "Etching",
        "Etching and drypoint on paper": "Etching",
        "Etching and engraving on paper": "Etching",
        "Etching and mezzotint on paper": "Etching",
        "Etching and watercolour on paper": "Etching",
        "Etching on paper": "Etching",
        "Oil paint": "Oil",
        "Oil paint on board": "Oil",
        "Oil paint on canvas": "Oil",
        "Oil paint on hardboard": "Oil",
        "Oil paint on mahogany": "Oil",
        "Oil paint on paper": "Oil",
        "Oil paint on wood": "Oil",
        "Ink": "Ink",
        "Ink and graphite on paper": "Ink",
        "Ink and watercolour on paper": "Ink",
        "Ink on paper": "Ink",
        "Ink wash and watercolour on paper": "Ink",
        "Wood": "Wood",
        "Wood engraving on paper": "Wood",
        "Woodcut on paper": "Wood",
        "Watercolour": "Watercolour",
        "Watercolour and gouache on paper": "Watercolour",
        "Watercolour and graphite on paper": "Graphite",
        "Watercolour and ink on paper": "Watercolour",
        "Watercolour on paper": "Watercolour",
        "Digital print on paper": "Print",
        "Photograph": "Photo",
        "Gouache": "Gouache",
        "Gouache and graphite on paper": "Gouache",
        "Gouache and watercolour on paper": "Gouache",
        "Gouache on paper": "Gouache",
        "Chalk": "Chalk",
        "Chalk and graphite on paper": "Chalk",
        "Chalk and watercolour on paper": "Chalk",
        "Chalk on paper": "Chalk",     
    }

def download_image(url, output_dir):
    try:
        filename = os.path.join(output_dir, os.path.basename(url))
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
        return filename
    except Exception:
        return None

def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_dataset(dataframe, batch_size, shuffle=True):
    file_paths = dataframe['local_path'].values
    labels = dataframe['encoded_label'].values

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))    
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=2000)        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds
def plot_split_confusion_matrix(cm, class_names, output_dir, max_classes_per_plot=20):
    n_classes = len(class_names)
    n_splits = (n_classes + max_classes_per_plot - 1) // max_classes_per_plot
    
    for split_idx in range(n_splits):
        start_idx = split_idx * max_classes_per_plot
        end_idx = min((split_idx + 1) * max_classes_per_plot, n_classes)
        
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            cm[start_idx:end_idx, start_idx:end_idx],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names[start_idx:end_idx],
            yticklabels=class_names[start_idx:end_idx],
            annot_kws={"size": 8}
        )
        plt.title(f'Confusion Matrix (Classes {start_idx}-{end_idx-1})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        split_path = os.path.join(output_dir, f"confusion_matrix_part_{split_idx+1}.png")
        plt.savefig(split_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(split_path)
        plt.close()

def build_model(num_classes, unfreeze_layers=None):
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ])
    x = augmentation(inputs)
    
    x = layers.Lambda(lambda img: (img - 0.5) / 0.5)(x)
    x = layers.Lambda(lambda img: tf.transpose(img, [0, 3, 1, 2]))(x)
    
    vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224")
    
    if unfreeze_layers is not None:
        vit_model.trainable = False
        
        if unfreeze_layers == "all":
            vit_model.trainable = True
        elif isinstance(unfreeze_layers, int):
            for layer in vit_model.layers[-unfreeze_layers:]:
                layer.trainable = True
        elif isinstance(unfreeze_layers, list):
            for idx in unfreeze_layers:
                vit_model.layers[idx].trainable = True
    
    vit_output = vit_model(pixel_values=x)
    x = vit_output.last_hidden_state[:, 0, :]
    
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

with mlflow.start_run(run_name=RUN_NAME):
    try:
        logger.info("Loading data")
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
        
        if MEMORY_SAVE_MODE:
            df = df[['thumbnailUrl', 'medium']]
        
        df.dropna(subset=['thumbnailUrl', 'medium'], inplace=True)        
        df['medium'] = df['medium'].apply(
            lambda x: str(x).split(',')[0].split('.')[0].strip()
        )
        
        class_counts = df['medium'].value_counts()
        classes_to_keep = class_counts[class_counts >= 50].index
        classes_to_keep = [c for c in classes_to_keep if c != "Video"]
        df = df[df['medium'].isin(classes_to_keep)]
        
        df['broad_category'] = df['medium'].map(CATEGORY_MAPPING)
        df['broad_category'] = df['broad_category'].fillna('Other')
        df['hierarchical_label'] = df['broad_category'] + " - " + df['medium']
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("Downloading images")

        df['local_path'] = None

        chunk_size = 1000
        total_rows = len(df)
        downloaded_count = 0

        for i in range(0, total_rows, chunk_size):
            chunk_end = min(i + chunk_size, total_rows)
            logger.info(f"Processing rows {i+1}-{chunk_end}/{total_rows}")
            
            for idx in range(i, chunk_end):
                url = df.iloc[idx]['thumbnailUrl']
                local_path = download_image(url, OUTPUT_DIR)
                df.iloc[idx, df.columns.get_loc('local_path')] = local_path
            
            if (i // chunk_size) % 5 == 0:
                progress_path = f"{OUTPUT_DIR}/temp_progress_{i}.csv"
                df.iloc[:chunk_end].to_csv(progress_path, index=False)
                logger.info(f"Saved progress to {progress_path}")
                gc.collect()

        initial_count = len(df)
        df.dropna(subset=['local_path'], inplace=True)
        downloaded_count = len(df)
        logger.info(f"Downloaded {downloaded_count}/{initial_count} images successfully")

        for f in os.listdir(OUTPUT_DIR):
            if f.startswith("temp_progress_"):
                os.remove(os.path.join(OUTPUT_DIR, f))
        
        logger.info("Preparing datasets")
        
        label_encoder = LabelEncoder()
        df['encoded_label'] = label_encoder.fit_transform(df['hierarchical_label'])
        num_classes = len(label_encoder.classes_)
        
        train_df, test_df = train_test_split(
            df,
            test_size=0.1,
            random_state=42,
            stratify=df['hierarchical_label']
        )        
        if MEMORY_SAVE_MODE:
            del df
            gc.collect()
        
        logger.info("Creating training dataset")
        train_ds = create_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True)
        
        logger.info("Creating test dataset")
        test_ds = create_dataset(test_df, batch_size=BATCH_SIZE, shuffle=False)
        
        if MEMORY_SAVE_MODE:
            del train_df, test_df
            gc.collect()
        
        logger.info("Calculating class weights")

        if 'train_df' in locals() or 'train_df' in globals():
            class_counts = train_df['encoded_label'].value_counts().sort_index()
            total_samples = len(train_df)
        else:
            logger.warning("Training DataFrame not available, estimating class weights")
            class_counts = np.bincount(
                np.concatenate([y.numpy() for x, y in train_ds], axis=0)
            )
            total_samples = sum(class_counts)

        class_weights = {}
        for class_index, count in enumerate(class_counts):
            if count > 0:
                class_weights[class_index] = total_samples / (len(class_counts) * count)
            else:
                class_weights[class_index] = 1.0

        logger.info(f"Class weights calculated for {len(class_weights)} classes")
        mlflow.log_param("class_weights", class_weights)
        
        class MetricsLogger(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.current_phase = 1
                self.phase_offset = 0

            def on_epoch_end(self, epoch, logs=None):
                epoch += self.phase_offset
                logs = logs or {}
                
                mlflow.log_metric("train_loss", logs.get("loss", 0), step=epoch)
                mlflow.log_metric("val_loss", logs.get("val_loss", 0), step=epoch)
                mlflow.log_metric("train_accuracy", logs.get("accuracy", 0), step=epoch)
                mlflow.log_metric("val_accuracy", logs.get("val_accuracy", 0), step=epoch)
                
                try:
                    if isinstance(self.model.optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                        current_step = self.model.optimizer.iterations.numpy()
                        lr_value = self.model.optimizer.learning_rate(current_step).numpy()
                        mlflow.log_metric("learning_rate", float(lr_value), step=epoch)
                    else:
                        lr_value = tf.keras.backend.get_value(self.model.optimizer.lr)
                        mlflow.log_metric("learning_rate", float(lr_value), step=epoch)
                except Exception as e:
                    logger.warning(f"Could not log learning rate: {e}")
                    mlflow.log_metric("learning_rate", 0.0, step=epoch)
                
                mlflow.log_metric("training_phase", self.current_phase, step=epoch)

        metrics_logger = MetricsLogger()

        logger.info("PHASE 1")
        model = build_model(num_classes, unfreeze_layers=0)

        steps_per_epoch = len(train_ds)
        total_steps_phase1 = EPOCHS_PHASE1 * steps_per_epoch

        lr_schedule_phase1 = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=3e-4,
            decay_steps=total_steps_phase1,
            alpha=0.05
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_phase1)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        callbacks_phase1 = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1
            ),
            metrics_logger
        ]
        history_phase1 = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=EPOCHS_PHASE1,
            callbacks=callbacks_phase1,
            class_weight=class_weights
        )
        phase1_weights_path = os.path.join(OUTPUT_DIR, "phase1_weights.h5")
        model.save_weights(phase1_weights_path)

        metrics_logger.current_phase = 2
        metrics_logger.phase_offset = len(history_phase1.history['loss'])

        logger.info("PHASE 2")

        total_steps_phase2 = EPOCHS_PHASE2 * steps_per_epoch

        lr_schedule_phase2 = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=5e-6,
            decay_steps=total_steps_phase2,
            alpha=0.05
        )
        optimizer_phase2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule_phase2)

        model = build_model(num_classes, unfreeze_layers=0)
        model.load_weights(phase1_weights_path)

        model.compile(
            optimizer=optimizer_phase2,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        unfreezing_schedule = [
            (2, 2),
            (4, 4),
            (6, 8),
            (8, 16),
            (12, "all")
        ]
        class GradualUnfreezeCallback(tf.keras.callbacks.Callback):
            def __init__(self, unfreezing_schedule, vit_model):
                super().__init__()
                self.unfreezing_schedule = unfreezing_schedule
                self.vit_model = vit_model
                self.epochs_trained = 0
                
            def on_epoch_end(self, epoch, logs=None):
                self.epochs_trained += 1
                for schedule_epoch, unfreeze_count in self.unfreezing_schedule:
                    if self.epochs_trained == schedule_epoch:
                        if unfreeze_count == "all":
                            for layer in self.vit_model.layers:
                                layer.trainable = True
                            logger.info(f"Epoch {self.epochs_trained}: Unfroze all layers")
                        else:
                            for layer in self.vit_model.layers[-unfreeze_count:]:
                                layer.trainable = True
                            logger.info(f"Epoch {self.epochs_trained}: Unfroze last {unfreeze_count} layers")

        vit_model = model.get_layer('tf_vi_t_model_1')
        unfreeze_callback = GradualUnfreezeCallback(unfreezing_schedule, vit_model)

        callbacks_phase2 = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1
            ),
            unfreeze_callback,
            metrics_logger
        ]

        history_phase2 = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=EPOCHS_PHASE2,
            callbacks=callbacks_phase2,
            class_weight=class_weights
        )

        
        logger.info("Evaluating model")
        test_loss, test_accuracy = model.evaluate(test_ds)

        mlflow.log_metric("final_test_accuracy", test_accuracy)
        mlflow.log_metric("final_test_loss", test_loss)

        y_pred = []
        for images, _ in test_ds:
            batch_pred = model.predict(images, verbose=0)
            y_pred.extend(batch_pred)
        y_pred = np.array(y_pred)

        y_true = np.concatenate([y for _, y in test_ds], axis=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        report = classification_report(
            y_true,
            y_pred_classes,
            target_names=label_encoder.classes_,
            output_dict=True
        )

        with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
            json.dump(report, f)
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "classification_report.json"))
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred_classes)
        class_names = label_encoder.classes_
        plot_split_confusion_matrix(cm, class_names, OUTPUT_DIR, max_classes_per_plot=15)
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        conf_matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(conf_matrix_path, bbox_inches='tight', dpi=300)
        mlflow.log_artifact(conf_matrix_path)
        plt.close()

        np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), cm)
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"))

        for class_name in report:
            if class_name in label_encoder.classes_:
                mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
                mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
                mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])

        logger.info("Saving model weights")
        weights_path = os.path.join(RUN_OUTPUT_DIR, "model_weights.h5")
        model.save_weights(weights_path)
        
        model_definition = f"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import TFViTModel

def build_model(num_classes={num_classes}, unfreeze_layers=None):
    inputs = tf.keras.Input(shape={IMAGE_SIZE + (3,)})
    
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ])
    x = augmentation(inputs)
    
    x = layers.Lambda(lambda img: (img - 0.5) / 0.5)(x)
    x = layers.Lambda(lambda img: tf.transpose(img, [0, 3, 1, 2]))(x)
    
    vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224")
    
    if unfreeze_layers is not None:
        vit_model.trainable = False
        
        if unfreeze_layers == "all":
            vit_model.trainable = True
        elif isinstance(unfreeze_layers, int):
            for layer in vit_model.layers[-unfreeze_layers:]:
                layer.trainable = True
        elif isinstance(unfreeze_layers, list):
            for idx in unfreeze_layers:
                vit_model.layers[idx].trainable = True
    
    vit_output = vit_model(pixel_values=x)
    x = vit_output.last_hidden_state[:, 0, :]
    
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)
"""
        model_def_path = os.path.join(RUN_OUTPUT_DIR, "model_definition.py")
        with open(model_def_path, 'w') as f:
            f.write(model_definition)
        
        classes_path = os.path.join(RUN_OUTPUT_DIR, "label_encoder_classes.npy")
        np.save(classes_path, label_encoder.classes_)
        
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "phase1_epochs": EPOCHS_PHASE1,
            "phase2_epochs": EPOCHS_PHASE2,
            "num_classes": num_classes,
            "memory_save_mode": MEMORY_SAVE_MODE
        })
        
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        
        mlflow.log_artifacts(RUN_OUTPUT_DIR)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        mlflow.log_param("error", str(e))
        raise