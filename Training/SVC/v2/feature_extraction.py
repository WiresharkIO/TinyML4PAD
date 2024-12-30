import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation
from sklearn.preprocessing import StandardScaler
from scipy import signal, stats
import os
import csv
from tqdm import tqdm
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.fftpack import rfft, rfftfreq

from numpy import cumsum, concatenate, zeros, linspace, average, power, absolute, mean, std, max, array, diff, where
from scipy.fftpack import fft, ifft
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import iqr
import scipy
import numpy
from scipy.stats import entropy
from definitions import ROOT_DIR


sourceFolder = ROOT_DIR
dataFolder_inlab_L = sourceFolder + '/datasets/inlab/left/'
dataFolder_inlab_R = sourceFolder + '/datasets/inlab/right/'
dataFolder_freeliving_L = sourceFolder + '/datasets/freeliving/'
featuresFolder = sourceFolder + '//features//sequence//preprocessing//feature_extracted//'


participants = ['01', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

Fs = 50
segment_length = 2.56
overlap = 0.5
frame_annotation_threshold = 0.3
frame_length = int(segment_length * Fs)
iter_length = frame_length * (1 - overlap)
window = np.hanning(frame_length)

def load_participant_data(participant_id, inlab_left_folder, inlab_right_folder, freeliving_folder):
    participant_data = {
        'inlab_left': None,
        'inlab_right': None,
        'freeliving_left': None
    }

    inlab_left = os.path.join(inlab_left_folder, f'participant_data_{participant_id}_left.csv')
    if os.path.exists(inlab_left):
        df_inlab_left = pd.read_csv(inlab_left)
        participant_data['inlab_left'] = {
            'prox_data': df_inlab_left['L_prox_sen_data'].values,
            'sequence_annotation': np.where(df_inlab_left['L_chew_cycle_annotation'].values > 0, 1, 0),
            'eating_annotation': [1 if x == 10000 else x for x in df_inlab_left['Eating_event_truth'].values]
        }

    inlab_right = os.path.join(inlab_right_folder, f'participant_data_{participant_id}_right.csv')
    if os.path.exists(inlab_right):
        df_inlab_right = pd.read_csv(inlab_right)
        participant_data['inlab_right'] = {
            'prox_data': df_inlab_right['R_prox_sen_data'].values,
            'sequence_annotation': np.where(df_inlab_right['R_chew_cycle_annotation'].values > 0, 1, 0),
            'eating_annotation': [1 if x == 10000 else x for x in df_inlab_right['Eating_event_truth'].values]
        }

    freeliving_left = os.path.join(freeliving_folder, f'participant_data_{participant_id}_left.csv')
    if os.path.exists(freeliving_left):
        df_freeliving_left = pd.read_csv(freeliving_left)
        participant_data['freeliving_left'] = {
            'prox_data': df_freeliving_left['L_prox_sen_data'].values,
            'sequence_annotation': df_freeliving_left['Annotation_Sequence_A'].values,
            'eating_annotation': [1 if x == 10000 else x for x in df_freeliving_left['Eating_event_truth'].values]
        }

    return participant_data

def combine_participant_sources(data):
    all_prox_data = []
    all_sequence_annotations = []
    all_eating_annotations = []

    for source, source_data in data.items():
        if source_data is not None:
            all_prox_data.extend(source_data['prox_data'])
            all_sequence_annotations.extend(source_data['sequence_annotation'])
            all_eating_annotations.extend(source_data['eating_annotation'])

    prox_data = np.array(all_prox_data)
    sequence_annotation = np.array(all_sequence_annotations)
    eating_annotation = np.array(all_eating_annotations)

    return prox_data, sequence_annotation, eating_annotation

def ema_filter(frame, beta=0.55):
    filtered = np.zeros_like(frame)
    filtered[0] = frame[0]

    for i in range(1, len(frame)):
        filtered[i] = filtered[i - 1] - (beta * (filtered[i - 1] - frame[i]))
    return filtered

def standardize_frame(filtered_frame):
    mean = np.mean(filtered_frame)
    std = np.std(filtered_frame)
    if std != 0:
        standardized_frame = (filtered_frame - mean) / std
    else:
        standardized_frame = filtered_frame - mean

    return standardized_frame

def plot_first_frame(prox_data, participant_id, beta=0.55):
    frame = prox_data[:128]  # Get first frame

    # Apply EMA filter
    filtered_frame = np.zeros_like(frame)
    filtered_frame[0] = frame[0]

    for i in range(1, len(frame)):
        filtered_frame[i] = filtered_frame[i - 1] - (beta * (filtered_frame[i - 1] - frame[i]))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(frame, 'b-', label='Raw Signal', alpha=0.7)
    plt.plot(filtered_frame, 'r-', label='EMA Filtered', linewidth=2)
    plt.title(f'First Frame - Participant {participant_id}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_crossing_patterns(standardized_frame):
    crossings = []
    crossing_amplitudes = []

    for i in range(1, len(standardized_frame)):
        if (standardized_frame[i - 1] * standardized_frame[i]) < 0:
            crossings.append(i)
            # Calculate absolute amplitude change at crossing
            amplitude = abs(standardized_frame[i] - standardized_frame[i - 1])
            crossing_amplitudes.append(amplitude)

    if len(crossings) > 1:
        intervals = np.diff(crossings)
        crossing_amplitudes = np.array(crossing_amplitudes)

        chew_lower = 13.1 - 8.64  # ≈ 4.5 samples
        chew_upper = 13.1 + 8.64  # ≈ 21.7 samples
        non_chew_upper = 2.89 + 3.96  # ≈ 6.85 samples

        chewing_intervals = (intervals >= chew_lower) & (intervals <= chew_upper)
        non_chewing_intervals = intervals <= non_chew_upper

        chewing_amplitudes = crossing_amplitudes[:-1][chewing_intervals]
        non_chewing_amplitudes = crossing_amplitudes[:-1][non_chewing_intervals]

        features = {
            'total_crossings': len(crossings),
            'mean_interval': np.mean(intervals),

            'mean_amplitude': np.mean(crossing_amplitudes),
            'max_amplitude': np.max(crossing_amplitudes),
            'min_amplitude': np.min(crossing_amplitudes),
            'std_amplitude': np.std(crossing_amplitudes),

            'chewing_mean_amp': np.mean(chewing_amplitudes) if len(chewing_amplitudes) > 0 else 0,
            'non_chewing_mean_amp': np.mean(non_chewing_amplitudes) if len(non_chewing_amplitudes) > 0 else 0,
            'chewing_std_amp': np.std(chewing_amplitudes) if len(chewing_amplitudes) > 0 else 0,
            'non_chewing_std_amp': np.std(non_chewing_amplitudes) if len(non_chewing_amplitudes) > 0 else 0,

            'chewing_intervals': np.sum(chewing_intervals),
            'non_chewing_intervals': np.sum(non_chewing_intervals),
            'chewing_ratio': np.sum(chewing_intervals) / len(intervals),
            'non_chewing_ratio': np.sum(non_chewing_intervals) / len(intervals),

            'is_chewing_range': np.mean(intervals) > non_chew_upper,
            'is_non_chewing_range': np.mean(intervals) <= non_chew_upper
        }

        return features

    return {
        'total_crossings': 0,
        'mean_interval': 0,
        'mean_amplitude': 0,
        'max_amplitude': 0,
        'min_amplitude': 0,
        'std_amplitude': 0,
        'chewing_mean_amp': 0,
        'non_chewing_mean_amp': 0,
        'chewing_std_amp': 0,
        'non_chewing_std_amp': 0,
        'chewing_intervals': 0,
        'non_chewing_intervals': 0,
        'chewing_ratio': 0,
        'non_chewing_ratio': 0,
        'is_chewing_range': False,
        'is_non_chewing_range': True
    }

def process_data_with_features(participant_id, prox_data, sequence_annotation, eating_annotation):
    resultFile = featuresFolder + 'sequence_features_' + participant_id + '.csv'
    # plot_first_frame(prox_data, participant_id)
    dataLength = len(prox_data)
    print('The data length is ' + str(dataLength) + ': 0-' + str(dataLength - 1))
    frame_index = int(dataLength // iter_length) - 1
    print('The frame index is ' + str(frame_index))
	
    # Initialize feature arrays
    signal_frame = np.zeros((frame_index, frame_length))

    ZC_TOTAL = np.zeros(frame_index)
    ZC_MEAN_INTERVAL = np.zeros(frame_index)
    ZC_CHEW_INTERVALS = np.zeros(frame_index)
    ZC_NONCHEW_INTERVALS = np.zeros(frame_index)
    ZC_CHEW_RATIO = np.zeros(frame_index)
    ZC_NONCHEW_RATIO = np.zeros(frame_index)
    ZC_IS_CHEW_RANGE = np.zeros(frame_index)
    ZC_IS_NONCHEW_RANGE = np.zeros(frame_index)

    # Amplitude feature arrays
    ZC_MEAN_AMP = np.zeros(frame_index)
    ZC_MAX_AMP = np.zeros(frame_index)
    ZC_MIN_AMP = np.zeros(frame_index)
    ZC_STD_AMP = np.zeros(frame_index)
    ZC_CHEW_MEAN_AMP = np.zeros(frame_index)
    ZC_NONCHEW_MEAN_AMP = np.zeros(frame_index)
    ZC_CHEW_STD_AMP = np.zeros(frame_index)
    ZC_NONCHEW_STD_AMP = np.zeros(frame_index)

    # Time-Domain Features
    TD_MAX = np.zeros(frame_index)
    TD_MIN = np.zeros(frame_index)
    TD_MAX_MIN = np.zeros(frame_index)
    TD_RMS = np.zeros(frame_index)
    TD_MEDIAN = np.zeros(frame_index)
    TD_VARIANCE = np.zeros(frame_index)
    TD_STD = np.zeros(frame_index)
    TD_SKEW = np.zeros(frame_index)
    TD_KURT = np.zeros(frame_index)
    TD_IQR = np.zeros(frame_index)

    # Ground Truth
    GROUND_TRUTH_FRAME = np.zeros(frame_index)
    EATING_TRUTH_FRAME = np.zeros(frame_index)
    SEQUENCE_TRUTH_FRAME = np.zeros(frame_index)
    BEGIN_INDEX_FRAME = np.zeros(frame_index)
    END_INDEX_FRAME = np.zeros(frame_index)

    # Process frames
    for i in tqdm(range(0, frame_index), desc="Processing frames"):
        low_count = int(i * iter_length)
        high_count = int(low_count + frame_length)

        frame = prox_data[low_count:high_count]
        filtered_frame = ema_filter(frame, beta=0.55)
        standardized_frame = standardize_frame(filtered_frame)
        # signal_frame[i, :] = filtered_frame

        zc_features = analyze_crossing_patterns(standardized_frame)
        # interval features
        ZC_TOTAL[i] = zc_features['total_crossings']
        ZC_MEAN_INTERVAL[i] = zc_features['mean_interval']
        ZC_CHEW_INTERVALS[i] = zc_features['chewing_intervals']
        ZC_NONCHEW_INTERVALS[i] = zc_features['non_chewing_intervals']
        ZC_CHEW_RATIO[i] = zc_features['chewing_ratio']
        ZC_NONCHEW_RATIO[i] = zc_features['non_chewing_ratio']
        ZC_IS_CHEW_RANGE[i] = zc_features['is_chewing_range']
        ZC_IS_NONCHEW_RANGE[i] = zc_features['is_non_chewing_range']

        # amplitude features
        ZC_MEAN_AMP[i] = zc_features['mean_amplitude']
        ZC_MAX_AMP[i] = zc_features['max_amplitude']
        ZC_MIN_AMP[i] = zc_features['min_amplitude']
        ZC_STD_AMP[i] = zc_features['std_amplitude']
        ZC_CHEW_MEAN_AMP[i] = zc_features['chewing_mean_amp']
        ZC_NONCHEW_MEAN_AMP[i] = zc_features['non_chewing_mean_amp']
        ZC_CHEW_STD_AMP[i] = zc_features['chewing_std_amp']
        ZC_NONCHEW_STD_AMP[i] = zc_features['non_chewing_std_amp']

        # TO-DO: change to filtered_frame
        TD_MAX[i] = np.max(standardized_frame)
        TD_MIN[i] = np.min(standardized_frame)
        TD_MAX_MIN[i] = TD_MAX[i] - TD_MIN[i]
        TD_RMS[i] = np.sqrt(np.mean(standardized_frame ** 2))
        TD_MEDIAN[i] = np.median(standardized_frame)
        TD_VARIANCE[i] = np.var(standardized_frame)
        TD_STD[i] = np.std(standardized_frame)
        TD_SKEW[i] = stats.skew(standardized_frame)
        TD_KURT[i] = stats.kurtosis(standardized_frame)
        TD_IQR[i] = stats.iqr(standardized_frame)

        # Ground Truth
        if sum(sequence_annotation[low_count:high_count]) > (frame_length * frame_annotation_threshold):
            SEQUENCE_TRUTH_FRAME[i] = 1
        else:
            SEQUENCE_TRUTH_FRAME[i] = 0

        if sum(eating_annotation[low_count:high_count]) > (frame_length * frame_annotation_threshold):
            EATING_TRUTH_FRAME[i] = 1
            GROUND_TRUTH_FRAME[i] = 1
        else:
            EATING_TRUTH_FRAME[i] = 0
            GROUND_TRUTH_FRAME[i] = 0

        BEGIN_INDEX_FRAME[i] = low_count
        END_INDEX_FRAME[i] = high_count

    feature_data = pd.DataFrame()
    feature_data['ZC_TOTAL'] = ZC_TOTAL

    feature_data['ZC_MEAN_INTERVAL'] = ZC_MEAN_INTERVAL
    feature_data['ZC_CHEW_INTERVALS'] = ZC_CHEW_INTERVALS
    feature_data['ZC_NONCHEW_INTERVALS'] = ZC_NONCHEW_INTERVALS
    feature_data['ZC_CHEW_RATIO'] = ZC_CHEW_RATIO
    feature_data['ZC_NONCHEW_RATIO'] = ZC_NONCHEW_RATIO
    feature_data['ZC_IS_CHEW_RANGE'] = ZC_IS_CHEW_RANGE
    feature_data['ZC_IS_NONCHEW_RANGE'] = ZC_IS_NONCHEW_RANGE

    feature_data['ZC_MEAN_AMP'] = ZC_MEAN_AMP
    feature_data['ZC_MAX_AMP'] = ZC_MAX_AMP
    feature_data['ZC_MIN_AMP'] = ZC_MIN_AMP
    feature_data['ZC_STD_AMP'] = ZC_STD_AMP
    feature_data['ZC_CHEW_MEAN_AMP'] = ZC_CHEW_MEAN_AMP
    feature_data['ZC_NONCHEW_MEAN_AMP'] = ZC_NONCHEW_MEAN_AMP
    feature_data['ZC_CHEW_STD_AMP'] = ZC_CHEW_STD_AMP
    feature_data['ZC_NONCHEW_STD_AMP'] = ZC_NONCHEW_STD_AMP

    feature_data['TD_MAX'] = TD_MAX
    feature_data['TD_MIN'] = TD_MIN
    feature_data['TD_MAX_MIN'] = TD_MAX_MIN
    feature_data['TD_RMS'] = TD_RMS
    feature_data['TD_MEDIAN'] = TD_MEDIAN
    feature_data['TD_VARIANCE'] = TD_VARIANCE
    feature_data['TD_STD'] = TD_STD
    feature_data['TD_SKEW'] = TD_SKEW
    feature_data['TD_KURT'] = TD_KURT
    feature_data['TD_IQR'] = TD_IQR

    feature_data['GROUND_TRUTH_FRAME'] = GROUND_TRUTH_FRAME
    feature_data['EATING_TRUTH_FRAME'] = EATING_TRUTH_FRAME
    feature_data['SEQUENCE_TRUTH_FRAME'] = SEQUENCE_TRUTH_FRAME
    feature_data['BEGIN_INDEX_FRAME'] = BEGIN_INDEX_FRAME
    feature_data['END_INDEX_FRAME'] = END_INDEX_FRAME

    feature_data.fillna(0)

    feature_data.to_csv(resultFile)
    print('Result saved to: ' + resultFile)

for participant_id in participants:
    print(f"Processing participant {participant_id}")

    participant_data = load_participant_data(
        participant_id,
        dataFolder_inlab_L,
        dataFolder_inlab_R,
        dataFolder_freeliving_L
    )

    prox_data, sequence_annotation ,eating_annotation = combine_participant_sources(participant_data)

    process_data_with_features(participant_id, prox_data, sequence_annotation, eating_annotation)

    print(f"Participant {participant_id} processed successfully")