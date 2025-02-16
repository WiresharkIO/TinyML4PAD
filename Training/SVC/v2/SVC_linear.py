import os
from definitions import ROOT_DIR

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score, f1_score
)


class LOPOFeatureSelectionSVC:
    def __init__(self, data_path, models_path, features_list=None):
        self.data_path = data_path
        self.models_path = models_path
        self.features_list = features_list
        os.makedirs(models_path, exist_ok=True)

        if features_list is None:

            self.features_list = ['ZC_TOTAL', 'ZC_CHEW_INTERVALS', 'ZC_NONCHEW_INTERVALS',
                                  'ZC_CHEW_RATIO',
                                  'TD_MAX', 'TD_MIN', 'TD_SKEW', 'TD_KURT']

    def load_and_prepare_data(self, participant_id, features_folder):

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        feature_files = [f for f in os.listdir(features_folder) if f.endswith('.csv')]

        for file in feature_files:
            file_path = os.path.join(features_folder, file)
            if f'sequence_features_{participant_id}.csv' == file:
                test_df = pd.read_csv(file_path)
                test_df = test_df.fillna(0)

            else:
                curr_df = pd.read_csv(file_path)
                curr_df = curr_df.fillna(0)
                train_df = pd.concat([train_df, curr_df], ignore_index=True)

        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")

        if not test_df.empty:
            print("\nClass distribution in test set:")
            print(test_df['SEQUENCE_TRUTH_FRAME'].value_counts(normalize=True))

        if not train_df.empty:
            print("\nClass distribution in training set:")
            print(train_df['SEQUENCE_TRUTH_FRAME'].value_counts(normalize=True))

        return train_df, test_df

    def _plot_confusion_matrix(self, tn, fp, fn, tp, participant_id):

        cm = np.array([[tn, fp], [fn, tp]])

        plt.figure(figsize=(10, 8))

        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix\nParticipant {participant_id}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.subplot(1, 2, 2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues')
        plt.title(f'Normalized Confusion Matrix\nParticipant {participant_id}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(os.path.join(self.models_path, f'confusion_matrix_{participant_id}.png'))
        plt.close()

    def evaluate_model(self, y_true, y_pred, participant_id):

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),

            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': f1_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'g_mean': np.sqrt((tp / (tp + fn)) * (tn / (tn + fp))) if (tp + fn) * (tn + fp) > 0 else 0,
            'precision_class_0': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'precision_class_1': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall_class_0': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'recall_class_1': tp / (tp + fn) if (tp + fn) > 0 else 0
        }

        self._plot_confusion_matrix(tn, fp, fn, tp, participant_id)

        return metrics

    def plot_metrics_comparison(self, results_df):
        """
        :param results_df:
        :return plot:

        'accuracy' - Less reliable for imbalanced data.

        'balanced_accuracy' - Accounts for imbalance by averaging recall for each class.

        'f1_score' - Harmonic mean of precision and recall.

        'mcc' - Matthews Correlation Coefficient, account true/false positives/negatives in a balanced way.
                Range: [-1, 1] where:
                    1: Perfect prediction
                    0: Random prediction
                    -1: Total disagreement

        'g_mean' - Geometric mean of precision and recall,
                   Good for measuring how well model performs on both classes.

        'precision' - Proportion of true positive predictions among all positive predictions.

        'recall' - Proportion of true positive predictions among all actual positive instances.

        'specificity' - Proportion of true negative predictions among all actual negative instances.

        """
		
        metrics_to_plot = ['accuracy', 'balanced_accuracy', 'f1_score',
                           'mcc', 'g_mean', 'precision', 'recall', 'specificity']

        fig, ax = plt.subplots(figsize=(15, 8))

        bar_width = 0.8 / len(metrics_to_plot)
        x = np.arange(len(results_df))

        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_to_plot)))

        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            positions = x + i * bar_width
            ax.bar(positions,
                   results_df[metric],
                   bar_width,
                   label=metric,
                   color=color,
                   alpha=0.8)

            metric_mean = results_df[metric].mean()
            ax.hlines(y=metric_mean,
                      xmin=x[0] + i * bar_width,
                      xmax=x[-1] + i * bar_width,
                      colors=color,
                      linestyles='dashed',
                      alpha=1.0,
                      linewidth=1.5)

            ax.text(x[-1] + i * bar_width + 0.1,
                    metric_mean,
                    f'avg: {metric_mean:.2f}',
                    color=color,
                    fontsize=8,
                    va='center')

            for j, v in enumerate(results_df[metric]):
                ax.text(positions[j],
                        v + 0.01,
                        f'{v:.2f}',
                        ha='center',
                        va='bottom',
                        rotation=90,
                        fontsize=8,
                        color=color)

        ax.set_xlabel('Participants', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance Metrics Comparison Across Participants\nwith Average Lines',
                     fontsize=14, pad=20)
        ax.set_xticks(x + (len(metrics_to_plot) - 1) * bar_width / 2)
        ax.set_xticklabels(results_df['participant_id'], rotation=0)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        ymax = max([results_df[metric].max() for metric in metrics_to_plot])
        ax.set_ylim(0, ymax * 1.15)

        plt.tight_layout()

        plt.savefig(os.path.join(self.models_path, 'metrics_comparison_with_averages.png'),
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

    def train_and_evaluate(self, participants):
        results = []

        for participant_id in tqdm(participants, desc="Processing participants"):
            print(f"\nProcessing participant {participant_id}")

            train_df, test_df = self.load_and_prepare_data(participant_id, self.data_path)

            X_train = train_df[self.features_list].values
            y_train = train_df['SEQUENCE_TRUTH_FRAME'].values
            X_test = test_df[self.features_list].values
            y_test = test_df['SEQUENCE_TRUTH_FRAME'].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            svc = LinearSVC(C=0.1, dual="auto", loss='squared_hinge',
                            penalty='l2', class_weight={0: 2, 1: 6},
                            max_iter=1000, verbose=True)

            svc.fit(X_train_scaled, y_train)

            y_pred = svc.predict(X_test_scaled)
            metrics = self.evaluate_model(y_test, y_pred, participant_id)

            metrics['participant_id'] = participant_id
            results.append(metrics)

            print(f"\nResults for participant {participant_id}:")
            print("\nPerformance Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
            print(f"F1 Score: {metrics['f1_score']:.3f}")
            print(f"MCC: {metrics['mcc']:.3f}")
            print(f"G-Mean: {metrics['g_mean']:.3f}")
            print("\nPer-class Metrics:")
            print(f"Class 0 - Precision: {metrics['precision_class_0']:.3f}, Recall: {metrics['recall_class_0']:.3f}")
            print(f"Class 1 - Precision: {metrics['precision_class_1']:.3f}, Recall: {metrics['recall_class_1']:.3f}")

        results_df = pd.DataFrame(results)
        self.plot_metrics_comparison(results_df)

        print("\nOverall Results:")
        for metric in ['accuracy', 'balanced_accuracy', 'f1_score', 'mcc', 'g_mean']:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"Average {metric}: {mean_val:.3f} ± {std_val:.3f}")

        return results_df


if __name__ == "__main__":
    sourceFolder = ROOT_DIR
    featuresFolder = sourceFolder + '//features//sequence//preprocessing//feature_extracted//v1//'
    savedModels = sourceFolder + '//models//sequence//SVM//training_results//metric_visualization_v1//'

    participants = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    lopo_svc = LOPOFeatureSelectionSVC(featuresFolder, savedModels)
    results = lopo_svc.train_and_evaluate(participants)