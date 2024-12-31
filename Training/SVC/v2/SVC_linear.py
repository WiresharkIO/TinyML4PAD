import os
from definitions import ROOT_DIR

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# import shap
from mrmr import mrmr_classif
from sklearn.feature_selection import RFE


class LOPOFeatureSelectionSVC:
    def __init__(self, data_path, models_path, features_list=None):
        self.data_path = data_path
        self.models_path = models_path
        self.features_list = features_list
        os.makedirs(models_path, exist_ok=True)
		
		# features used..
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
        train_data_participants=[]
        test_data_participants=[]
        for file in feature_files:
            file_path = os.path.join(features_folder, file)

            if f'sequence_features_{participant_id}.csv' == file:
                test_df = pd.read_csv(file_path)
                test_df = test_df.fillna(0)
                test_data_participants.append(participant_id)
            else:
                curr_df = pd.read_csv(file_path)
                curr_df = curr_df.fillna(0)
                train_df = pd.concat([train_df, curr_df], ignore_index=True)
                train_data_participants.append(participant_id)

		# checks
        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
		
		# checks
        print(f"Training set participants: {train_data_participants}")
        print(f"Test set participants: {test_data_participants}")


        if not test_df.empty:
            print("\nClass distribution in test set:")
            print(test_df['SEQUENCE_TRUTH_FRAME'].value_counts(normalize=True))

        if not train_df.empty:
            print("\nClass distribution in training set:")
            print(train_df['SEQUENCE_TRUTH_FRAME'].value_counts(normalize=True))

        return train_df, test_df

    def select_features(self, X_train, y_train, X_test, n_features=20):

        # 1. RFE-Linear SVC
        svc = LinearSVC(class_weight='balanced', dual=False)
        rfe = RFE(estimator=svc, n_features_to_select=n_features)
        rfe.fit(X_train, y_train)
        rfe_support = rfe.support_

        # 2. SVM Weighting
        svc.fit(X_train, y_train)
        svm_weights = np.abs(svc.coef_[0])
        svm_importance = np.argsort(-svm_weights)[:n_features]

        # 3. mRMR
        X_train_df = pd.DataFrame(X_train, columns=self.features_list)
        selected_features = mrmr_classif(X=X_train_df, y=pd.Series(y_train), K=n_features)

        # 4. SHAP
        # explainer = shap.LinearExplainer(svc, X_train)
        # shap_values = explainer.shap_values(X_train)
        # shap_importance = np.argsort(-np.abs(shap_values).mean(0))[:n_features]

        feature_votes = np.zeros(len(self.features_list))
        feature_votes[rfe_support] += 1
        feature_votes[svm_importance] += 1
        # feature_votes[shap_importance] += 1

        for feature in selected_features:
            feature_votes[self.features_list.index(feature)] += 1

        selected_indices = np.argsort(-feature_votes)[:n_features]

        X_train_reduced = X_train[:, selected_indices]
        X_test_reduced = X_test[:, selected_indices]

        selected_feature_names = [self.features_list[i] for i in selected_indices]
        print("Selected features:", selected_feature_names)

        return X_train_reduced, X_test_reduced, selected_indices

    def get_class_weights(self, y):

        classes = np.unique(y)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y
        )

        class_weight_dict = dict(zip(classes, weights))

        print("Computed class weights:", class_weight_dict)
        return class_weight_dict

    def evaluate_participant(self, y_true, y_pred, participant_id):

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Participant {participant_id}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.models_path, f'confusion_matrix_{participant_id}.png'))
        plt.close()

        return accuracy, precision, recall, f1

    def train_and_evaluate(self, participants):
        results = []
        for participant_id in participants:

            print(f"\nProcessing participant {participant_id}")

            train_df, test_df = self.load_and_prepare_data(participant_id, self.data_path)

            X_train = train_df[self.features_list].values
            y_train = train_df['SEQUENCE_TRUTH_FRAME'].values

            X_test = test_df[self.features_list].values
            y_test = test_df['SEQUENCE_TRUTH_FRAME'].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            X_train_reduced, X_test_reduced, selected_features = self.select_features(
                X_train_scaled, y_train, X_test_scaled
            )

            # class_weights = self.get_class_weights(y_train)
            """svm_V1"""
            # svc = LinearSVC(
            #     C=1.0,
            #     dual="auto",
            #     loss='squared_hinge',
            #     penalty='l2',
            #     class_weight={0: 1, 1: 15},
            #     max_iter=20000,
            #     verbose=True
            # )
            """
            svm_V2          
            """
            """  - Support Vectors Classifier tries to find the best hyperplane to separate the different 
                   classes by maximizing the distance between sample points and the hyperplane.
                   
                 - penalty='l2'Mathematical form: ||w||₂² = w₁² + w₂² + ... + wₙ²..
				 - the dual should be 'false since n_features<n_samples, but 'auto' works..
            """

            svc = LinearSVC(C=0.1, dual="auto", loss='squared_hinge', penalty='l2', class_weight={0: 2, 1: 6},
                            max_iter=1000, verbose=True)
            svc.fit(X_train_reduced, y_train)

            y_pred = svc.predict(X_test_reduced)
            accuracy, precision, recall, f1 = self.evaluate_participant(y_test, y_pred, participant_id)

            results.append({
                'participant_id': participant_id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

            # model and scaler
            joblib.dump(svc, os.path.join(self.models_path, f'svc_model_{participant_id}.joblib'))
            joblib.dump(scaler, os.path.join(self.models_path, f'scaler_{participant_id}.joblib'))

            # selected features
            selected_features_dict = {
                'features': [self.features_list[i] for i in selected_features]
            }
            joblib.dump(
                selected_features_dict,
                os.path.join(self.models_path, f'selected_features_{participant_id}.joblib')
            )

            print(f"Results for participant {participant_id}:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"F1 Score: {f1:.3f}")
            print(f"recall: {recall:.3f}")
            print(f"precision: {precision:.3f}")

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
