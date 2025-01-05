import os
from definitions import ROOT_DIR

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# import shap
from mrmr import mrmr_classif
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance


class LOPOFeatureSelectionSVC:
    def __init__(self, data_path, models_path, features_list=None):
        self.data_path = data_path
        self.models_path = models_path
        self.features_list = features_list
        os.makedirs(models_path, exist_ok=True)

        if features_list is None:
            self.features_list = ['ZC_TOTAL', 'ZC_MEAN_INTERVAL', 'ZC_CHEW_INTERVALS', 'ZC_NONCHEW_INTERVALS',
                                  'ZC_CHEW_RATIO', 'ZC_NONCHEW_RATIO', 'ZC_IS_CHEW_RANGE', 'ZC_IS_NONCHEW_RANGE',
                                  'ZC_MEAN_AMP', 'ZC_MAX_AMP', 'ZC_MIN_AMP', 'ZC_STD_AMP',
                                  'ZC_CHEW_MEAN_AMP', 'ZC_NONCHEW_MEAN_AMP', 'ZC_CHEW_STD_AMP', 'ZC_NONCHEW_STD_AMP',

                                  'TD_MAX', 'TD_MIN', 'TD_MAX_MIN', 'TD_RMS', 'TD_MEDIAN',
                                  'TD_VARIANCE', 'TD_STD', 'TD_SKEW', 'TD_KURT', 'TD_IQR']

    def load_and_prepare_data(self, participant_id, features_folder):

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        feature_files = [f for f in os.listdir(features_folder) if f.endswith('.csv')]
        train_data_participants = []
        test_data_participants = []
        for file in feature_files:
            file_path = os.path.join(features_folder, file)

            if f'sequence_features_{participant_id}.csv' == file:
                test_df = pd.read_csv(file_path)
                test_df = test_df.fillna(0)
                test_data_participants.append(file)
            else:
                curr_df = pd.read_csv(file_path)
                curr_df = curr_df.fillna(0)
                train_df = pd.concat([train_df, curr_df], ignore_index=True)
                train_data_participants.append(file)

        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")

        print(f"Training set participants: {train_data_participants}")
        print(f"Test set participants: {test_data_participants}")

        if not test_df.empty:
            print("\nClass distribution in test set:")
            print(test_df['SEQUENCE_TRUTH_FRAME'].value_counts(normalize=True))

        if not train_df.empty:
            print("\nClass distribution in training set:")
            print(train_df['SEQUENCE_TRUTH_FRAME'].value_counts(normalize=True))

        return train_df, test_df

    def select_features_pre_training(self, X_train, y_train, X_test, n_features=20):

        svc = LinearSVC(C=10000, dual="auto", loss='squared_hinge',
                        penalty='l2', class_weight={0: 2, 1: 6},
                        max_iter=1000)

        rfe = RFE(estimator=svc, n_features_to_select=n_features)
        rfe.fit(X_train, y_train)
        rfe_support = rfe.support_

        svc.fit(X_train, y_train)
        svm_weights = np.abs(svc.coef_[0])

        # Normalizing SVM weights to [0,1] range..
        svm_weights = (svm_weights - np.min(svm_weights)) / (np.max(svm_weights) - np.min(svm_weights))

        X_train_df = pd.DataFrame(X_train, columns=self.features_list)
        selected_features = mrmr_classif(X=X_train_df, y=pd.Series(y_train), K=n_features)

        feature_votes = np.zeros(len(self.features_list))
        feature_votes[rfe_support] += 1

        feature_votes += svm_weights

        for feature in selected_features:
            feature_votes[self.features_list.index(feature)] += 1

        selected_indices = np.argsort(-feature_votes)[:n_features]
        return selected_indices, feature_votes

    def select_features_post_training(self, model, X_train, X_test, y_test, selected_indices, n_features=15):

        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]

        perm_importance = permutation_importance(model, X_test_selected, y_test, n_repeats=10)
        perm_scores = perm_importance.importances_mean
        perm_scores = (perm_scores - np.min(perm_scores)) / (np.max(perm_scores) - np.min(perm_scores))

        coef_importance = np.abs(model.coef_[0])
        coef_importance = (coef_importance - np.min(coef_importance)) / (
                    np.max(coef_importance) - np.min(coef_importance))

        cv_scores = np.zeros(len(selected_indices))
        for i in range(len(selected_indices)):
            X_single = X_test_selected[:, [i]]
            cv_score = cross_val_score(model, X_single, y_test, cv=5).mean()
            cv_scores[i] = cv_score
        cv_scores = (cv_scores - np.min(cv_scores)) / (np.max(cv_scores) - np.min(cv_scores))

        post_training_scores = (perm_scores + coef_importance + cv_scores) / 3
        final_indices = selected_indices[np.argsort(-post_training_scores)[:n_features]]

        return final_indices, post_training_scores

    def get_class_weights(self, y, participant_id):

        classes = np.unique(y)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y
        )

        class_weight_dict = dict(zip(classes, weights))
        print(f"IGNORE THIS WEIGHTS FOR NOW for {participant_id}, {class_weight_dict}")

        # return class_weight_dict

    def evaluate_participant(self, y_true, y_pred, participant_id):

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

        return accuracy, precision, recall, f1

    def plot_feature_importance_comparison(self, participant_id, pre_features, post_features):
	
        plt.figure(figsize=(15, 10))

        # for pre-training importance
        plt.subplot(2, 1, 1)
        features_pre, scores_pre = zip(*sorted(pre_features, key=lambda x: x[1], reverse=True))
        bars_pre = plt.bar(range(len(features_pre)), scores_pre, color='skyblue')
        plt.title(f'Pre-training Feature Importance - Participant {participant_id}')
        plt.xticks(range(len(features_pre)), features_pre, rotation=45, ha='right')
        plt.ylabel('Importance Score')

        for bar in bars_pre:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom')

        # for post-training importance
        plt.subplot(2, 1, 2)
        features_post, scores_post = zip(*sorted(post_features, key=lambda x: x[1], reverse=True))
        bars_post = plt.bar(range(len(features_post)), scores_post, color='lightcoral')
        plt.title(f'Post-training Feature Importance - Participant {participant_id}')
        plt.xticks(range(len(features_post)), features_post, rotation=45, ha='right')
        plt.ylabel('Importance Score')

        for bar in bars_post:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.models_path, f'feature_importance_{participant_id}.png'))
        plt.close()

    def plot_feature_stability(self, results_df):
        feature_counts = {}
        for _, row in results_df.iterrows():
            for feature in row['final_features']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        features, counts = zip(*sorted_features)

        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(features)), counts, color='lightseagreen')
        plt.title('Feature Selection Stability Across Participants')
        plt.xlabel('Features')
        plt.ylabel('Number of Participants')
        plt.xticks(range(len(features)), features, rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     int(height),
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.models_path, 'feature_stability.png'))
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

            # just checks - not used in training..
            self.get_class_weights(y_train, participant_id)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            selected_indices, pre_importance = self.select_features_pre_training(
                X_train_scaled, y_train, X_test_scaled
            )

            X_train_selected = X_train_scaled[:, selected_indices]
            svc = LinearSVC(C=10000, dual="auto", loss='squared_hinge',
                            penalty='l2', class_weight={0: 2, 1: 6},
                            max_iter=1000, verbose=True)
            svc.fit(X_train_selected, y_train)

            final_indices, post_importance = self.select_features_post_training(
                svc, X_train_scaled, X_test_scaled, y_test, selected_indices
            )

            X_train_final = X_train_scaled[:, final_indices]
            X_test_final = X_test_scaled[:, final_indices]

            final_model = LinearSVC(C=10000, dual="auto", loss='squared_hinge',
                                    penalty='l2', class_weight={0: 2, 1: 6},
                                    max_iter=1000, verbose=True)
            final_model.fit(X_train_final, y_train)

            y_pred = final_model.predict(X_test_final)
            accuracy, precision, recall, f1 = self.evaluate_participant(
                y_test, y_pred, participant_id
            )

            print("\nPre-training Feature Importance:")
            pre_features = [(self.features_list[i], pre_importance[i])
                            for i in selected_indices]
            for feature, score in sorted(pre_features, key=lambda x: x[1], reverse=True):
                print(f"{feature}: {score:.3f}")

            print("\nPost-training Feature Importance:")
            indices_map = {idx: pos for pos, idx in enumerate(selected_indices)}
            post_features = [(self.features_list[i], post_importance[indices_map[i]])
                             for i in final_indices]
            for feature, score in sorted(post_features, key=lambda x: x[1], reverse=True):
                print(f"{feature}: {score:.3f}")

            self.plot_feature_importance_comparison(participant_id, pre_features, post_features)

            results.append({
                'participant_id': participant_id,
                'pre_features': [self.features_list[i] for i in selected_indices],
                'final_features': [self.features_list[i] for i in final_indices],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
			
        self.plot_feature_stability(pd.DataFrame(results))
        return pd.DataFrame(results)


if __name__ == "__main__":
    sourceFolder = ROOT_DIR
    featuresFolder = sourceFolder + '//features//sequence//preprocessing//feature_extracted//'
    savedModels = sourceFolder + '//models//sequence//SVM//inference//'

    participants = ["01", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]

    lopo_svc = LOPOFeatureSelectionSVC(featuresFolder, savedModels)
    results = lopo_svc.train_and_evaluate(participants)

    print("\nOverall Results:")
    print(f"Average Accuracy: {results['accuracy'].mean():.3f} ± {results['accuracy'].std():.3f}")
    print(f"Average F1 Score: {results['f1_score'].mean():.3f} ± {results['f1_score'].std():.3f}")
    print(f"Average Recall: {results['recall'].mean():.3f} ± {results['recall'].std():.3f}")
    print(f"Average Precision: {results['precision'].mean():.3f} ± {results['precision'].std():.3f}")

