import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score)
from imblearn.combine import SMOTEENN
from datetime import datetime
from definitions import ROOT_DIR
from os import walk
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sys import platform
from joblib import Parallel, delayed
import traceback
import shap
import os

sourceFolder = ROOT_DIR
modelFolder = sourceFolder + '\\models\\sequence\\saved_models\\'

isExist = os.path.exists(modelFolder)
if not isExist:
    os.makedirs(modelFolder)

featureFolder = sourceFolder + '\\models\\sequence\\chew_dataset_3s.csv'


def call_svm():
    print("Start SVM training at: " + str(datetime.now()))
    featuresDataframe = pd.read_csv(featureFolder)
    featuresDataframe.fillna(0)

    # features_list = ['TD_MAX', 'TD_MIN', 'TD_MAX_MIN', 'TD_RMS', 'TD_MEDIAN', 'TD_VARIANCE', 'TD_STD', 'TD_SKEW',
    #                  'TD_KURT', 'TD_IQR', 'FD_MEAN', 'FD_POWB', 'FD_MEDIAN', 'TFD_MIN', 'TFD_MAX', 'TFD_PSD_MEAN',
    #                  'TFD_STD', 'TFD_S_ENT', 'TFD_S_KURT', 'TFD_KURT', 'TFD_SKEW', 'TFD_PSD_MEDIAN', 'TFD_AMP_KURT',
    #                  'TFD_AMP_SKEW', 'TFD_ERG_SUM', 'TFD_ERG_MIN', 'TFD_ERG_MAX', 'TFD_ERG_MEAN', 'TFD_ERG_Q1',
    #                  'TFD_ERG_Q2', 'TFD_ERG_Q3', 'TFD_ERG_Q4', 'TFD_ERG_1', 'TFD_ERG_2', 'TFD_ERG_3', 'TFD_ERG_4',
    #                  'TFD_ERG_5', 'TFD_ERG_6', 'TFD_ERG_7', 'TFD_CONCENTRATION']

    # features_list = ['TFD_ERG_Q1', 'TFD_ERG_MIN', 'TFD_MIN', 'TFD_AMP_SKEW', 'TD_KURT', 'TFD_ERG_Q2', 'TFD_ERG_SUM',
    #                  'TFD_SKEW', 'TFD_PSD_MEDIAN', 'TFD_AMP_KURT', 'TFD_ERG_MEAN', 'TFD_STD', 'TFD_S_ENT', 'TFD_KURT',
    #                  'TD_IQR', 'TFD_S_KURT', 'FD_MEDIAN', 'TFD_ERG_MAX', 'TFD_MAX', 'FD_POWB']
    features_list = ['TFD_S_ENT', 'TD_KURT', 'TFD_STD', 'TFD_MAX', 'TFD_KURT', 'FD_MEDIAN', 'TFD_S_KURT']
    # print(len(features_list)) --> 40

    label = ['SEQUENCE_TRUTH_FRAME']
    featuresArray = featuresDataframe[features_list].to_numpy()
    featuresArray[np.isnan(featuresArray)] = 0

    labelArray = featuresDataframe[label].to_numpy()
    labelArray = labelArray.flatten()

    X_train, X_test, y_train, y_test = train_test_split(featuresArray, labelArray, test_size=0.2,
                                                        random_state=42)
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(X_test)

    svc = LinearSVC(C=0.1, dual="auto", loss='squared_hinge', penalty='l2', class_weight={0: 2, 1: 5}, max_iter=15000,
                    verbose=True)
    svc.fit(X_train_standardized, y_train)

    try:

        joblib.dump(svc, modelFolder + 'fea_svcLinear.joblib')
        print("model saved as 'fea_svcLinear.joblib'")

        joblib.dump(scaler, modelFolder + 'fea_svcLinear_scaler.joblib')
        print("Scaler saved as 'fea_svcLinear_scaler.joblib'")

        y_pred = svc.predict(X_test_standardized)
        accuracy_overall = accuracy_score(y_test, y_pred)
        f1_overall = f1_score(y_test, y_pred)
        precision_overall = precision_score(y_test, y_pred)
        recall_overall = recall_score(y_test, y_pred)

        print(f"Accuracy: {accuracy_overall:.4f}")
        print(f"Precision: {f1_overall:.4f}")
        print(f"Recall: {precision_overall:.4f}")
        print(f"F1-score: {recall_overall:.4f}")

        ''' 1. Implementation of ROC curve'''
        # fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test_standardized))
        # optimal_idx = np.argmax(tpr - fpr)
        # optimal_threshold = thresholds[optimal_idx]
        #
        #
        # y_pred = (svc.decision_function(X_test_standardized) > optimal_threshold).astype(int)
        #
        # accuracy_overall = accuracy_score(y_test, y_pred)
        # f1_overall = f1_score(y_test, y_pred)
        # precision_overall = precision_score(y_test, y_pred)
        # recall_overall = recall_score(y_test, y_pred)
        #
        # print(f"Accuracy: {accuracy_overall:.4f}")
        # print(f"Precision: {f1_overall:.4f}")
        # print(f"Recall: {precision_overall:.4f}")
        # print(f"F1-score: {recall_overall:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(modelFolder + 'confusion_matrix_fea_svcLinear.png')
        # plt.show()
        plt.close()

        print("Ended SVM Training at: " + str(datetime.now()))

    except Exception as e:
        print("An error occurred during model training:")
        print(traceback.format_exc())
        print("Ended SVM Training at: " + str(datetime.now()))

    print('------------------Training-Ended----------------------')

if __name__ == '__main__':
    print('------------------Training-Begins----------------------')
    call_svm()