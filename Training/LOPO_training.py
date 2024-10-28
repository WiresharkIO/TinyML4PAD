import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
from os import walk
import joblib
from definitions import ROOT_DIR
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


def load_and_prepare_data(inlab_folder, freeliving_folder):

    inlab_files = next(walk(inlab_folder), (None, None, []))[2]
    freeliving_files = next(walk(freeliving_folder), (None, None, []))[2]

    return inlab_files, freeliving_files


def create_lopo_dataset(participant_id, inlab_files, freeliving_files, inlab_folder, freeliving_folder):

    train_df = pd.DataFrame()

    for inlab_file in inlab_files:
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


def train_and_evaluate_lopo():

    models_folder = sourceFolder + '\\models\\sequence\\LOPO_features\\LOPO_training\\'
    results_folder = sourceFolder + '\\models\\sequence\\LOPO_features\\LOPO_training\\inference\\'
    inlab_folder = sourceFolder + '\\features\\sequence\\inlab\\inlab_3s\\'
    freeliving_folder = sourceFolder + '\\features\\sequence\\freeliving\\freeliving_3s\\'

    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    inlab_files, freeliving_files = load_and_prepare_data(inlab_folder, freeliving_folder)
    # print(inlab_files, freeliving_files)

    # features_list = ['TD_MAX', 'TD_MIN', 'TD_MAX_MIN', 'TD_RMS', 'TD_MEDIAN', 'TD_VARIANCE', 'TD_STD', 'TD_SKEW',
    #                  'TD_KURT', 'TD_IQR',
    #
    #                  'FD_MEDIAN',
    #
    #                  'TFD_MAX', 'TFD_STD', 'TFD_S_ENT', 'TFD_S_KURT', 'TFD_KURT', 'TFD_AMP_SKEW']

    features_list = ['TD_KURT', 'FD_MEDIAN', 'TFD_MAX', 'TFD_STD', 'TFD_S_ENT', 'TFD_S_KURT', 'TFD_KURT']

    all_results = []

    for participant_id in freeliving_files:
        print(f"\nFor participant: {participant_id}")

        train_df, test_df = create_lopo_dataset(
            participant_id, inlab_files, freeliving_files,
            inlab_folder, freeliving_folder
        )

        X_train, y_train = prepare_features_and_labels(train_df, features_list)
        X_test, y_test = prepare_features_and_labels(test_df, features_list)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svc = LinearSVC(C=10000, dual="auto", loss='squared_hinge', penalty='l2', class_weight={0: 2, 1: 6},
                        max_iter=1000, verbose=True)

        svc.fit(X_train_scaled, y_train)

        y_pred = svc.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        results = {
            'participant_id': participant_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        all_results.append(results)
        match = re.search(r'_(\d{2})_', participant_id)
        if match:
            extracted_id = match.group(1)
        else:
            raise ValueError("Could not extract participant ID from filename")
        # model_filename = os.path.join(models_folder, f'LOPO_model_3s_{extracted_id}.joblib')
        # scaler_filename = os.path.join(models_folder, f'LOPO_scaler_3s_{extracted_id}.joblib')
        # joblib.dump(svc, model_filename)
        # joblib.dump(scaler, scaler_filename)

        print(f"Results for {participant_id}:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")

    results_df = pd.DataFrame(all_results)
    # results_df.to_csv(os.path.join(results_folder, 'lopo_results.csv'), index=False)

    print("\nOverall Results:")
    print(f"Average Accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
    print(f"Average F1 Score: {results_df['f1_score'].mean():.3f} ± {results_df['f1_score'].std():.3f}")

    return results_df


def plot_lopo_performance(results_df):

    participant_ids = sorted([int(re.search(r'_(\d{2})_', id).group(1)) for id in results_df['participant_id']])

    accuracies = results_df['accuracy'].values
    f1_scores = results_df['f1_score'].values

    avg_accuracy = np.mean(accuracies)
    avg_f1_score = np.mean(f1_scores)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('LOPO Model Performance', fontsize=16, fontweight='bold')

    individual_color = 'xkcd:azul'
    average_color = 'xkcd:battleship grey'

    ax1.plot(participant_ids, accuracies, color=individual_color, marker='o', linestyle='-', linewidth=2, markersize=8)
    ax1.axhline(y=avg_accuracy, color=average_color, linestyle='--', linewidth=2)

    ax1.fill_between(participant_ids, accuracies, avg_accuracy, where=(accuracies > avg_accuracy), interpolate=True,
                     alpha=0.3, color=individual_color)
    ax1.fill_between(participant_ids, accuracies, avg_accuracy, where=(accuracies <= avg_accuracy), interpolate=True,
                     alpha=0.3, color=average_color)

    ax1.set_title('Accuracy for Each LOPO Participant', fontsize=14)
    ax1.set_xlabel('Participant ID', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_ylim(min(accuracies) - 0.05, max(accuracies) + 0.05)

    ax2.plot(participant_ids, f1_scores, color=individual_color, marker='o', linestyle='-', linewidth=2, markersize=8)
    ax2.axhline(y=avg_f1_score, color=average_color, linestyle='--', linewidth=2)

    ax2.fill_between(participant_ids, f1_scores, avg_f1_score, where=(f1_scores > avg_f1_score), interpolate=True,
                     alpha=0.3, color=individual_color)
    ax2.fill_between(participant_ids, f1_scores, avg_f1_score, where=(f1_scores <= avg_f1_score), interpolate=True,
                     alpha=0.3, color=average_color)

    ax2.set_title('F1 Score for Each LOPO Participant', fontsize=14)
    ax2.set_xlabel('Participant ID', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_ylim(min(f1_scores) - 0.05, max(f1_scores) + 0.05)

    legend_elements = [
        Patch(facecolor=individual_color, edgecolor='black', label='Individual Participants'),
        Patch(facecolor=average_color, edgecolor='black', label='Overall Average')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=12)

    model_params = "Model Parameters:\n" \
                   "C=10000, dual='auto'\n" \
                   "loss='squared_hinge', penalty='l2'\n" \
                   "class_weight={0: 2, 1: 6}\n" \
                   "max_iter=1000"
    fig.text(0.02, 0.02, model_params, fontsize=10, verticalalignment='bottom')

    overall_metrics = f"Overall Performance:\n" \
                      f"Avg Accuracy: {avg_accuracy:.3f}\n" \
                      f"Avg F1 Score: {avg_f1_score:.3f}"
    fig.text(0.98, 0.02, overall_metrics, fontsize=10, horizontalalignment='right', verticalalignment='bottom')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)
    # plt.style.use('seaborn-darkgrid')
    plt.show()


if __name__ == "__main__":
    sourceFolder = ROOT_DIR

    results = train_and_evaluate_lopo()

    plot_lopo_performance(results)