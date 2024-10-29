import numpy as np
import pandas as pd
import joblib
import os
from os import walk
from definitions import ROOT_DIR

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def load_and_prepare_data(inlab_folder, freeliving_folder):

    inlab_files = next(walk(inlab_folder), (None, None, []))[2]
    freeliving_files = next(walk(freeliving_folder), (None, None, []))[2]

    return inlab_files, freeliving_files


def create_lopo_dataset(participant_id, inlab_files, freeliving_files, inlab_folder, freeliving_folder):

    train_df = pd.DataFrame()

    for inlab_file in inlab_files:
        if inlab_file[35:37] != participant_id[35:37]:
            df = pd.read_csv(os.path.join(inlab_folder, inlab_file))
            train_df = pd.concat([train_df, df], ignore_index=True)

    for freeliving_file in freeliving_files:
        if freeliving_file != participant_id:
            df = pd.read_csv(os.path.join(freeliving_folder, freeliving_file))
            train_df = pd.concat([train_df, df], ignore_index=True)

    test_df = pd.read_csv(os.path.join(freeliving_folder, participant_id))

    return train_df, test_df


def prepare_features_and_labels(df, features_list, label_column='SEQUENCE_TRUTH_FRAME'):

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    X = df[features_list]
    y = df[label_column]

    X = X.fillna(0)

    return X, y


def call_teacher_training(models_folder):

    inlab_folder = sourceFolder + '\\features\\sequence\\inlab\\inlab_2.56s\\'
    freeliving_folder = sourceFolder + '\\features\\sequence\\freeliving\\freeliving_2.56s\\'

    os.makedirs(models_folder, exist_ok=True)

    inlab_files, freeliving_files = load_and_prepare_data(inlab_folder, freeliving_folder)

    # features_list = ['TD_MAX', 'TD_MIN', 'TD_MAX_MIN', 'TD_RMS', 'TD_MEDIAN', 'TD_VARIANCE', 'TD_STD', 'TD_SKEW',
    #                  'TD_KURT', 'TD_IQR',
    #
    #                  'FD_MEDIAN',
    #
    #                  'TFD_MAX', 'TFD_STD', 'TFD_S_ENT', 'TFD_S_KURT', 'TFD_KURT', 'TFD_AMP_SKEW']

    features_list = ['TD_KURT', 'FD_MEDIAN', 'TFD_MAX', 'TFD_STD', 'TFD_S_ENT', 'TFD_S_KURT', 'TFD_KURT']

    model_files = [f for f in os.listdir(models_folder) if f.startswith('LOPO_model_2.56s_')]
    # print(model_files)
    models = [joblib.load(os.path.join(models_folder, model_file)) for model_file in sorted(model_files)]
    # print(model)
    scaler_files = [f for f in os.listdir(models_folder) if f.startswith('LOPO_scaler_2.56s_')]
    print(scaler_files)
    scalers = [joblib.load(os.path.join(models_folder, model_file)) for model_file in sorted(scaler_files)]
    print(scalers)

    # distillation data and soft labels..
    X_distillation = []
    soft_labels_distillation = []
    index=0
    for participant_id in freeliving_files:
        print(f"\nFor participant: {participant_id}")

        train_df, test_df = create_lopo_dataset(
            participant_id, inlab_files, freeliving_files,
            inlab_folder, freeliving_folder
        )

        X_train, y_train = prepare_features_and_labels(train_df, features_list)
        scaler = scalers[index]
        index+=1
        X_train_scaled = scaler.transform(X_train)

        ensemble_predictions = [model.decision_function(X_train_scaled) for model in models]
        ensemble_predictions = np.array(ensemble_predictions)
        # print(ensemble_predictions)

        soft_labels = np.mean(ensemble_predictions, axis=0)

        X_distillation.append(X_train_scaled)
        soft_labels_distillation.append(soft_labels)

    X_distillation = np.concatenate(X_distillation, axis=0)
    soft_labels_distillation = np.concatenate(soft_labels_distillation, axis=0)
    # print(X_distillation.shape, soft_labels_distillation.shape)

    return X_distillation, soft_labels_distillation, scalers


def call_student_training(X_distillation, soft_labels_distillation):

    student_svc = LinearSVC()
    threshold = 0.5
    discrete_labels = (soft_labels_distillation >= threshold).astype(int)
    student_svc.fit(X_distillation, discrete_labels)

    return student_svc


def test_student_model(student_model, inlab_folder, freeliving_folder, features_list, scaler):

    inlab_files, freeliving_files = load_and_prepare_data(inlab_folder, freeliving_folder)

    # Create a separate test set
    test_results = []
    index_scaler=0
    for participant_id in freeliving_files:
        test_df = pd.read_csv(os.path.join(freeliving_folder, participant_id))
        X_test, y_test = prepare_features_and_labels(test_df, features_list)
        scalers_student=scaler[index_scaler]
        index_scaler+=1

        X_test_scaled = scalers_student.transform(X_test)

        y_pred = student_model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        test_results.append({
            'participant': participant_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    results_df = pd.DataFrame(test_results)

    print("\nTest Results per Participant:")
    print(results_df)
    print("\nAverage Results:")
    print(results_df.mean(numeric_only=True))

    return results_df


def plot_student_performance(results_df):

    participant_ids = np.arange(1, len(results_df) + 1)

    accuracies = results_df['accuracy'].values
    f1_scores = results_df['f1'].values

    avg_accuracy = np.mean(accuracies)
    avg_f1_score = np.mean(f1_scores)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle('Student Model Performance Across Participants',
                 fontsize=16, fontweight='bold', y=0.95)

    individual_color = '#3498db'
    average_color = '#7f8c8d'
    # below_avg_color = '#87fd05'
    below_avg_color = '#a83c09'

    ax1.plot(participant_ids, accuracies, color=individual_color,
             marker='o', linestyle='-', linewidth=2, markersize=8)
    ax1.axhline(y=avg_accuracy, color=average_color, linestyle='--', linewidth=2)

    ax1.fill_between(participant_ids, accuracies, avg_accuracy,
                     where=(accuracies > avg_accuracy), interpolate=True,
                     alpha=0.3, color=individual_color)
    ax1.fill_between(participant_ids, accuracies, avg_accuracy,
                     where=(accuracies <= avg_accuracy), interpolate=True,
                     alpha=0.3, color=below_avg_color)

    ax1.set_xticks(participant_ids)
    ax1.set_xticklabels([f'P{i}' for i in participant_ids])

    ax1.set_title('Accuracy for Each Participant', fontsize=14, pad=20)
    ax1.set_xlabel('Participant ID', fontsize=12, labelpad=10)
    ax1.set_ylabel('Accuracy', fontsize=12, labelpad=10)
    ax1.grid(True, linestyle=':', alpha=0.7)

    accuracy_padding = (max(accuracies) - min(accuracies)) * 0.1
    ax1.set_ylim(min(accuracies) - accuracy_padding,
                 max(accuracies) + accuracy_padding)

    ax2.plot(participant_ids, f1_scores, color=individual_color,
             marker='o', linestyle='-', linewidth=2, markersize=8)
    ax2.axhline(y=avg_f1_score, color=average_color, linestyle='--', linewidth=2)

    ax2.set_xticks(participant_ids)
    ax2.set_xticklabels([f'P{i}' for i in participant_ids])

    ax2.fill_between(participant_ids, f1_scores, avg_f1_score,
                     where=(f1_scores > avg_f1_score), interpolate=True,
                     alpha=0.3, color=individual_color)
    ax2.fill_between(participant_ids, f1_scores, avg_f1_score,
                     where=(f1_scores <= avg_f1_score), interpolate=True,
                     alpha=0.3, color=below_avg_color)

    ax2.set_title('F1 Score for Each Participant', fontsize=14, pad=20)
    ax2.set_xlabel('Participant ID', fontsize=12, labelpad=10)
    ax2.set_ylabel('F1 Score', fontsize=12, labelpad=10)
    ax2.grid(True, linestyle=':', alpha=0.7)

    f1_padding = (max(f1_scores) - min(f1_scores)) * 0.1
    ax2.set_ylim(min(f1_scores) - f1_padding,
                 max(f1_scores) + f1_padding)

    legend_elements = [
        Patch(facecolor=individual_color, edgecolor='black',
              label='Individual Participants'),
        Patch(facecolor=average_color, edgecolor='black',
              label='Overall Average')
    ]

    ax1.legend(handles=legend_elements,
               loc='upper right',
               bbox_to_anchor=(1.15, 1.30),
               ncol=2,
               fontsize=12)

    param_ax = fig.add_axes([0.1, 0.02, 0.4, 0.05])
    param_ax.axis('off')
    model_params = "Student Model Parameters:\n" \
                   "Model: LinearSVC\n" \
                   "Trained with knowledge distillation\n" \
                   "from ensemble of LOPO models"
    param_ax.text(0, 0.5, model_params,
                  fontsize=10,
                  verticalalignment='center')

    metrics_ax = fig.add_axes([0.6, 0.02, 0.4, 0.05])
    metrics_ax.axis('off')
    overall_metrics = f"Overall Performance:\n" \
                      f"Avg Accuracy: {avg_accuracy:.3f}\n" \
                      f"Avg F1 Score: {avg_f1_score:.3f}"
    metrics_ax.text(1, 0.5, overall_metrics,
                    fontsize=10,
                    horizontalalignment='right',
                    verticalalignment='center')

    plt.subplots_adjust(top=0.85,  # Space at top for legend
                        bottom=0.15,  # Space at bottom for text
                        hspace=0.3)  # Space between subplots

    return fig


if __name__ == "__main__":

    sourceFolder = ROOT_DIR
    models_folder = sourceFolder + '\\models\\sequence\\LOPO_training'

    X_distillation, soft_labels_distillation, scalers = call_teacher_training(models_folder + '\\trained_model_files\\')
    student_model = call_student_training(X_distillation, soft_labels_distillation)

    features_list = ['TD_KURT', 'FD_MEDIAN', 'TFD_MAX', 'TFD_STD',
                     'TFD_S_ENT', 'TFD_S_KURT', 'TFD_KURT']

    test_results = test_student_model(
        student_model,
        inlab_folder=sourceFolder + '\\features\\sequence\\inlab\\inlab_2.56s\\',
        freeliving_folder=sourceFolder + '\\features\\sequence\\freeliving\\freeliving_2.56s\\',
        features_list=features_list,
        scaler=scalers
    )
    # print(test_results)

    fig = plot_student_performance(test_results)
    fig.savefig(models_folder + '\\training_results\\student_model_performance.png', bbox_inches='tight', dpi=300)

    joblib.dump(student_model, models_folder + "\\trained_model_files\\student_model.joblib")
    print("Student model saved as joblib format.")

    input_shape = X_distillation.shape[1]
    print(f"Input shape: {input_shape}")

    student_model = joblib.load(models_folder + "\\trained_model_files\\student_model.joblib")
    input_shape = X_distillation.shape[1]
    initial_type = [('input', FloatTensorType([None, input_shape]))]

    onnx_model = convert_sklearn(student_model, initial_types=initial_type)

    with open(models_folder + "\\inference\\student_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("Student model converted to ONNX format.")
