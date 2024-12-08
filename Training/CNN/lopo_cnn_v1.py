import numpy as np
import pandas as pd
import os
from os import walk
from definitions import ROOT_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score
from imblearn.metrics import specificity_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import keras
import tensorflow as tf
print(tf.__version__)
import joblib

from keras import mixed_precision
import onnx
import onnxruntime as ort
import tf2onnx
import json
import datetime
import matplotlib.pyplot as plt


def load_data(dataFolder):
    left_folder = os.path.join(dataFolder, 'left')
    right_folder = os.path.join(dataFolder, 'right')
    skip_participants = ['02', '14', '15']

    participant_data = {}

    for file in os.listdir(left_folder):
        if file.endswith('.csv'):
            participant_id = file[17:19]
            if participant_id not in skip_participants:
                df = pd.read_csv(os.path.join(left_folder, file))
                if participant_id not in participant_data:
                    participant_data[participant_id] = {'sensor_data': [], 'labels': []}
                participant_data[participant_id]['sensor_data'].extend(df['L_prox_sen_data'].values)
                # converting multi-class to binary labels
                binary_labels = np.where(df['L_chew_cycle_annotation'].values > 0, 1, 0)
                participant_data[participant_id]['labels'].extend(binary_labels)

    for file in os.listdir(right_folder):
        if file.endswith('.csv'):
            participant_id = file[17:19]
            if participant_id not in skip_participants:
                df = pd.read_csv(os.path.join(right_folder, file))
                if participant_id not in participant_data:
                    participant_data[participant_id] = {'sensor_data': [], 'labels': []}
                participant_data[participant_id]['sensor_data'].extend(df['R_prox_sen_data'].values)
                # converting multi-class to binary labels
                binary_labels = np.where(df['R_chew_cycle_annotation'].values > 0, 1, 0)
                participant_data[participant_id]['labels'].extend(binary_labels)

    return participant_data


def create_frame(data, labels, frame_size=128, overlap=64):
    frame = []
    frame_labels = []

    for i in range(0, len(data) - frame_size, overlap):
        window_data = data[i:i + frame_size]
        window_labels = labels[i:i + frame_size]

        window_label = np.mean(window_labels) >= 0.3

        if np.std(window_data) > 1e-6:
            frame.append(window_data)
            frame_labels.append(int(window_label))

    return np.array(frame), np.array(frame_labels)


def create_model(frame_size):
    model = keras.Sequential([
        keras.layers.Input(shape=(frame_size,)),
        keras.layers.Reshape((frame_size, 1)),

        keras.layers.Conv1D(32, kernel_size=32, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling1D(pool_size=4),

        keras.layers.Conv1D(64, kernel_size=16, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.GlobalAveragePooling1D(),

        keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def train_model(frame_size, X_train, y_train, X_val, y_val, test_participant, scaler):
    trained_model_files = sourceFolder + '\\models\\sequence\\LOPO_training\\inference\\LOPO_CNN_V2\\'
    model = create_model(frame_size)

    total_samples = len(y_train)
    n_samples_0 = np.sum(y_train == 0)
    n_samples_1 = np.sum(y_train == 1)

    weight_0 = total_samples / (2 * n_samples_0)
    weight_1 = total_samples / (2 * n_samples_1)

    class_weight_dict = {0: weight_0, 1: weight_1}

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        # Use focal loss instead of binary crossentropy
        loss=keras.losses.BinaryFocalCrossentropy(alpha=0.75, gamma=2.0),
        metrics=['accuracy', keras.metrics.AUC(),
                 keras.metrics.Precision(),
                 keras.metrics.Recall()]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,  
        batch_size=32, 
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    model_path = os.path.join(trained_model_files, f"participant_{test_participant}_model.keras")
    model.save(model_path)
    scaler_path = os.path.join(trained_model_files, f"participant_{test_participant}_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    return model, history


def evaluate_model(model, X_test, y_test, show_plots=True):
    y_pred = model.predict(X_test) > 0.5
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # balanced accuracy
    balanced_acc = (recall_score(y_test, y_pred) +
                    specificity_score(y_test, y_pred)) / 2

    metrics = {
        'accuracy': report['accuracy'],
        'balanced_accuracy': balanced_acc,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'auc': roc_auc_score(y_test, y_pred)
    }
    return metrics


def lopo_cross_validation(participant_data, frame_size=128, overlap=64):
    results = {}

    for test_participant in participant_data.keys():
        print(f"\nTraining model for participant {test_participant}")
        scaler = StandardScaler()

        train_sensor_data = []
        train_labels = []
        for participant_id, data in participant_data.items():
            if participant_id != test_participant:
                train_sensor_data.extend(data['sensor_data'])
                train_labels.extend(data['labels'])

        test_sensor_data = participant_data[test_participant]['sensor_data']
        test_labels = participant_data[test_participant]['labels']

        X_train, y_train = create_frame(np.array(train_sensor_data), np.array(train_labels), frame_size, overlap)
        X_test, y_test = create_frame(np.array(test_sensor_data), np.array(test_labels), frame_size, overlap)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model, _ = train_model(frame_size, X_train_scaled, y_train, X_test_scaled, y_test, test_participant, scaler)
        results[test_participant] = evaluate_model(model, X_test_scaled, y_test, show_plots=False)

    return results


def main():
    models_folder = sourceFolder + '\\models\\sequence\\LOPO_training\\'
    dataFolder = sourceFolder + '\\datasets\\inlab\\'
    trained_model_files = sourceFolder + '\\models\\sequence\\LOPO_training\\inference\\LOPO_CNN_V2\\'
    training_results = models_folder + '\\training_results\\'
    results_folder = sourceFolder + '\\models\\sequence\\LOPO_training\\inference\\'

    participant_data = load_data(dataFolder)
    results = lopo_cross_validation(participant_data)

    print("\nLOPO Cross-validation Results:")
    for participant, metrics in results.items():
        print(f"\nParticipant {participant}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")

    avg_accuracy = np.mean([m['accuracy'] for m in results.values()])
    avg_balanced_accuracy = np.mean([m['balanced_accuracy'] for m in results.values()])
    avg_precision = np.mean([m['precision'] for m in results.values()])
    avg_recall = np.mean([m['recall'] for m in results.values()])
    avg_f1 = np.mean([m['f1'] for m in results.values()])
    avg_auc = np.mean([m['auc'] for m in results.values()])

    print(f"\nAverage Results:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Balanced_accuracy: {avg_balanced_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1-score: {avg_f1:.4f}")
    print(f"AUC: {avg_auc:.4f}")

    for filename in os.listdir(trained_model_files):
        if filename.endswith("_model.keras"):
            participant_id = filename.split("_")[1]  # Extract participant ID
            model_path = os.path.join(trained_model_files, filename)

            model = keras.models.load_model(model_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            tflite_filename = f"participant_{participant_id}_model.tflite"
            tflite_path = os.path.join(trained_model_files, tflite_filename)

            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Converted and saved TFLite model for participant {participant_id}")


if __name__ == "__main__":
    sourceFolder = ROOT_DIR
    main()