import os
import urllib.request
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow
import json
import shutil
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# Configuration
DATA_PATH = 'D:/venv sa python 3.9/Diplomski/clean_output.csv'
#DATA_PATH = r"D:\venv sa python 3.9\Diplomski\clean_output - kratki testni.csv"
#OUTPUT_DIR = r"D:\venv sa python 3.9\Diplomski\testno"
OUTPUT_DIR = "D:/venv sa python 3.9/tate_images"
RUN_OUTPUT_DIR = "D:/venv sa python 3.9/Diplomski/runovi"
MLFLOW_URI = "http://127.0.0.1:5000"
RUN_NAME = "Tate_Collection_EfficientNetV1"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50 

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Tate_Collection1")

# Category Mapping
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
            logger.info(f"Downloaded: {url}")
        return filename
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return None

def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label

def create_dataset(file_paths, labels, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths))    
    ds = ds.batch(batch_size)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

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

def build_model(num_classes):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )    
    # Freeze base model layers
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# Main Execution
with mlflow.start_run(run_name=RUN_NAME):
    try:
        logger.info("Loading data")
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
        logger.info(f"Loaded data with shape: {df.shape}")
        
        df.dropna(subset=['thumbnailUrl', 'medium'], inplace=True)
        logger.info(f"Data after cleaning: {df.shape}")        
        df['medium'] = df['medium'].apply(
            lambda x: str(x).split(',')[0].split('.')[0].strip()
        )        
        # Filter classes
        class_counts = df['medium'].value_counts()
        classes_to_keep = class_counts[class_counts >= 50].index
        classes_to_keep = [c for c in classes_to_keep if c != "Video"]
        df = df[df['medium'].isin(classes_to_keep)]
        logger.info(f"Classes after filtering: {list(classes_to_keep)}")
        
        df['broad_category'] = df['medium'].map(CATEGORY_MAPPING)
        
        missing_categories = df[df['broad_category'].isna()]['medium'].unique()
        if len(missing_categories) > 0:
            logger.warning(f"Unmapped mediums: {missing_categories}")
            df['broad_category'] = df['broad_category'].fillna('Other')
        
        df['hierarchical_label'] = df['broad_category'] + " - " + df['medium']
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("Downloading images")        
        df['local_path'] = df['thumbnailUrl'].apply(
            lambda url: download_image(url, OUTPUT_DIR)
        )
        
        initial_count = len(df)
        df.dropna(subset=['local_path'], inplace=True)
        logger.info(f"Downloaded {len(df)}/{initial_count} images successfully")
        
        logger.info("Preparing datasets")        
        
        label_encoder = LabelEncoder()
        df['encoded_label'] = label_encoder.fit_transform(df['hierarchical_label'])
        num_classes = len(label_encoder.classes_)
        mlflow.log_param("num_classes", num_classes)
        
        train_df, test_df = train_test_split(
            df,
            test_size=0.1,
            random_state=42,
            stratify=df['hierarchical_label']
        )
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        train_ds = create_dataset(
            train_df['local_path'].values,
            train_df['encoded_label'].values,
            batch_size=BATCH_SIZE,
            shuffle=True
        )        
        test_ds = create_dataset(
            test_df['local_path'].values,
            test_df['encoded_label'].values,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        # 4. Calculate class weights
        logger.info("Calculating class weights")

        class_counts = train_df['encoded_label'].value_counts().sort_index()
        total_samples = len(train_df)

        class_weights = {}
        for class_index, count in class_counts.items():
            class_weights[class_index] = total_samples / (len(class_counts) * count)
            
        logger.info(f"Class weights: {class_weights}")
        mlflow.log_param("class_weights", class_weights)

        plt.figure(figsize=(12, 6))
        plt.bar(label_encoder.classes_, class_counts.values)
        plt.xticks(rotation=90)
        plt.title("Class Distribution")
        plt.tight_layout()
        dist_path = os.path.join(RUN_OUTPUT_DIR, "class_distribution.png")
        plt.savefig(dist_path)
        mlflow.log_artifact(dist_path)
    
        # 4. Build and train model
        logger.info("Building model")
        model = build_model(num_classes)        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )        
        model.summary(print_fn=logger.info)
        summary_string = []
        model.summary(print_fn=lambda x: summary_string.append(x))
        summary_text = "\n".join(summary_string)
        summary_path = os.path.join(RUN_OUTPUT_DIR, "model_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary_text)

        mlflow.log_artifact(summary_path)        
        logger.info("Training model")
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )        
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=EPOCHS,
            callbacks=[early_stopping],
            class_weight=class_weights
        )
        
        for epoch in range(len(history.history['accuracy'])):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], epoch)
            mlflow.log_metric("train_loss", history.history['loss'][epoch], epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], epoch)
        
        # 5. Evaluate model
        logger.info("Evaluating model")
        test_loss, test_accuracy = model.evaluate(test_ds)
        logger.info(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
        
        # Generate predictions
        y_pred = model.predict(test_ds)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.concatenate([y for _, y in test_ds], axis=0)
        
        # Classification report
        report = classification_report(
            y_true,
            y_pred_classes,
            target_names=label_encoder.classes_,
            output_dict=True
        ) 
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        
        report_path = os.path.join(RUN_OUTPUT_DIR, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_path)
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred_classes)
        class_names = label_encoder.classes_                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        plot_split_confusion_matrix(cm, class_names, RUN_OUTPUT_DIR, max_classes_per_plot=15)
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        conf_matrix_path = os.path.join(RUN_OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(conf_matrix_path, bbox_inches='tight', dpi=300)
        mlflow.log_artifact(conf_matrix_path)
        plt.close()

        np.save(os.path.join(RUN_OUTPUT_DIR, "confusion_matrix.npy"), cm)
        mlflow.log_artifact(os.path.join(RUN_OUTPUT_DIR, "confusion_matrix.npy"))
        for class_name in report:
            if class_name in label_encoder.classes_:
                mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
                mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
                mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])

        # 6. Save model (Weights-only)
        logger.info("Saving model weights")        
        weights_path = os.path.join(RUN_OUTPUT_DIR, "model_weights.h5")
        model.save_weights(weights_path, save_format='h5')       
        mlflow.log_artifact(weights_path)        
        
        model_definition = f"""
def build_model(num_classes={num_classes}):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape={IMAGE_SIZE + (3,)}
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape={IMAGE_SIZE + (3,)})
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)
"""
        model_def_path = os.path.join(RUN_OUTPUT_DIR, "model_definition.py")
        with open(model_def_path, 'w') as f:
            f.write(model_definition)
        mlflow.log_artifact(model_def_path)
        
        # Save label encoder classes
        classes_path = os.path.join(RUN_OUTPUT_DIR, "label_encoder_classes.npy")
        np.save(classes_path, label_encoder.classes_)
        mlflow.log_artifact(classes_path)                         
        
        samples = []
        for cls in classes_to_keep:
            cls_df = df[df['medium'] == cls]
            samples.append(cls_df.sample(n=min(10, len(cls_df)), random_state=42))
        
        sample_df = pd.concat(samples)
        sample_path = os.path.join(RUN_OUTPUT_DIR, "tate_samples.csv")
        sample_df.to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        mlflow.log_param("error", str(e))
        raise