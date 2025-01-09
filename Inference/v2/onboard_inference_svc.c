#include "main.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "arm_math.h"
#include "svm_features_01.h"
#include "test_data.h"

/* model file includes */
#include "svc_linear_one.h"
#include "svc_linear_one_data.h"
#include "svc_linear_one_data_params.h"

#include "ai_datatypes_defines.h"
#include "ai_platform.h"

/* signal processing parameters before feature extraction in raw data(signal) */
#define FS 50 /* sampling frequency i.e the number of samples per second */
#define SEGMENT_LENGTH 2.56f /* in seconds */
#define FRAME_LENGTH 128
#define OVERLAP 0.5f
#define ITER_LENGTH (FRAME_LENGTH * (1.0f - OVERLAP))

/* for EMA filter */
#define EMA_BETA 0.55f

/* for checks */
#define FIRST_ITERATION 1
#define NOT_FIRST_ITERATION 2

/* for true label calculation */
#define CHEW_THRESHOLD 0.3f

/* for zero crossing intervals */
#define CHEW_LOWER 4.5f
#define CHEW_UPPER 21.7f
#define NON_CHEW_UPPER 6.85f

/* for inference on-board */
/* for DMA handling */
/* this determines the buffer size, current capacity is 256-bytes */
#define DMA_BUFFER_SIZE 256
/* samples taken at a time */
#define NUM_FLOATS 64
#define TEST_LENGTH 320

/* data-gathering and storing in circular buffer to extract features for inference */
typedef struct {
    float data[FRAME_LENGTH];
    int head;
    int tail;
    int count;
} CircularBuffer;

enum{
    ZC_TOTAL_IDX = 0,
    ZC_CHEW_INTERVALS_IDX,
    ZC_NONCHEW_INTERVALS_IDX,
    ZC_CHEW_RATIO_IDX,
    TD_MAX_IDX,
    TD_MIN_IDX,
    TD_SKEW_IDX,
    TD_KURT_IDX
};

typedef struct {
    int true_positives;
    int false_positives;
    int false_negatives;
    int true_negatives;
    float precision;
    float recall;
    float f1_score;
    float accuracy;
} Metrics;

UART_HandleTypeDef hlpuart1;

CircularBuffer input_buffer;

/* model IO variables and buffers */
ai_handle proximity_data;
AI_ALIGNED(4) ai_float aiOutData[AI_SVC_LINEAR_ONE_OUT_1_SIZE_BYTES];
AI_ALIGNED(4) ai_float aiInData[AI_SVC_LINEAR_ONE_IN_1_SIZE];
ai_u8 activations[AI_SVC_LINEAR_ONE_DATA_ACTIVATIONS_SIZE];
ai_buffer *ai_input;
ai_buffer *ai_output;

/* for DMA handling */
uint8_t dma_buffer[DMA_BUFFER_SIZE];
volatile bool dma_rx_complete = false;

void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_LPUART1_UART_Init(void);

/* data-gathering and storing in circular buffer to extract features for inference */
void circular_buffer_init(CircularBuffer* cb);
void circular_buffer_push(CircularBuffer* cb, float item);
void circular_buffer_pop(CircularBuffer* cb, float* item);
void apply_ema_filter(const float32_t *input_frame, float32_t *output_frame, int frame_size);

/* model related function prototypes */
int get_frame_activity_class(int frame_start_index, const int* activity_class, int frame_length);
static void AI_Init(void);
static void AI_Run(float *pIn, float *pOut);

void standardize_filtered_frame(const float* input_frame, float* output_frame, int frame_length);
void standardize_features(const float* features, float* standardized_features);

void circular_buffer_init(CircularBuffer* cb) {
    cb->head = 0;
    cb->tail = 0;
    cb->count = 0;
}

void circular_buffer_push(CircularBuffer* cb, float item) {
    cb->data[cb->head] = item;
    cb->head = (cb->head + 1) % FRAME_LENGTH;
    if (cb->count < FRAME_LENGTH) {
        cb->count++;
    } else {
        cb->tail = (cb->tail + 1) % FRAME_LENGTH;
    }
}

void circular_buffer_pop(CircularBuffer* cb, float* item) {
    *item = cb->data[cb->tail];
    cb->tail = (cb->tail + 1) % FRAME_LENGTH;
    cb->count--;
}

void standardize_filtered_frame(const float32_t* input_frame, float32_t* output_frame, int frame_length) {
    float mean = 0.0f;
    for(int i = 0; i < frame_length; i++) {
        mean += input_frame[i];
    }
    mean /= frame_length;

    float variance = 0.0f;
    for(int i = 0; i < frame_length; i++) {
        float diff = input_frame[i] - mean;
        variance += diff * diff;
    }
    variance /= frame_length;
    float std = sqrtf(variance);

//    float32_t mean=0.0f, std=0.0f;
//	arm_mean_f32(input_frame, frame_length, &mean);
//	arm_std_f32(input_frame, frame_length, &std);

    if(std != 0) {
        for(int i = 0; i < frame_length; i++) {
            output_frame[i] = (input_frame[i] - mean) / std;
        }
    } else {

        for(int i = 0; i < frame_length; i++) {
            output_frame[i] = input_frame[i] - mean;
        }
    }
}

void standardize_features(const float* features, float32_t* standardized_features) {
    for(int i = 0; i < NUM_FEATURES; i++) {
        standardized_features[i] = (features[i] - feature_means[i]) / feature_scales[i];
    }
}

void apply_ema_filter(const float32_t *input_frame, float32_t *output_frame, int frame_size) {

    output_frame[0] = input_frame[0];

    for(int i = 1; i < frame_size; i++) {
        output_frame[i] = output_frame[i-1] - (EMA_BETA * (output_frame[i-1] - input_frame[i]));
    }
}

float extract_ZC_TOTAL(const float32_t* signal, int length) {
    float zc_total = 0;
    for(int i = 1; i < length; i++) {
        if((signal[i-1] * signal[i]) < 0) {
            zc_total += 1;
        }
    }
    return zc_total;
}

void analyze_intervals(const float32_t* signal, int length, float* chew_intervals, float* nonchew_intervals, float* chew_ratio) {
	int crossing_count = 0;
    int crossing_indices[128];

    for(int i = 1; i < length; i++) {
        if((signal[i-1] * signal[i]) < 0) {
            if(crossing_count < 128) {
                crossing_indices[crossing_count] = i;
                crossing_count++;
            }
        }
    }

    *chew_intervals = 0;
    *nonchew_intervals = 0;
    if(crossing_count > 1) {
    	int total_intervals = crossing_count - 1;
		for(int i = 1; i < crossing_count; i++) {
			int interval = crossing_indices[i] - crossing_indices[i-1];
			if(interval >= CHEW_LOWER && interval <= CHEW_UPPER) {
				(*chew_intervals)++;
			}
			if(interval <= NON_CHEW_UPPER) {
				(*nonchew_intervals)++;
			}
		}
		*chew_ratio = (total_intervals > 0) ? ((float)*chew_intervals / total_intervals) : 0.0f;
    }
    else
    {
		*chew_ratio = 0.0f;
	}
}

float extract_ZC_CHEW_INTERVALS(const float32_t* signal, int length) {
    float chew_intervals, nonchew_intervals, chew_ratio;
    analyze_intervals(signal, length, &chew_intervals, &nonchew_intervals, &chew_ratio);
    return chew_intervals;
}

float extract_ZC_NONCHEW_INTERVALS(const float32_t* signal, int length) {
    float chew_intervals, nonchew_intervals, chew_ratio;

    analyze_intervals(signal, length, &chew_intervals, &nonchew_intervals, &chew_ratio);
    return nonchew_intervals;
}

float extract_ZC_CHEW_RATIO(const float32_t* signal, int length) {
    float chew_intervals, nonchew_intervals, chew_ratio;

    analyze_intervals(signal, length, &chew_intervals, &nonchew_intervals, &chew_ratio);

    return chew_ratio;
}

float extract_TD_MAX(const float32_t* signal, int length) {
    float max = signal[0];
    for(int i = 1; i < length; i++) {
        if(signal[i] > max) max = signal[i];
    }
    return max;
}

float extract_TD_MIN(const float32_t* signal, int length) {
    float min = signal[0];
    for(int i = 1; i < length; i++) {
        if(signal[i] < min) min = signal[i];
    }
    return min;
}

float calculate_mean(const float32_t* signal, int length) {
    float sum = 0;
    for(int i = 0; i < length; i++) {
        sum += signal[i];
    }
    return sum / length;
}

float calculate_std(const float32_t* signal, float mean, int length) {
    float sum_squared_diff = 0;
    for(int i = 0; i < length; i++) {
        float diff = signal[i] - mean;
        sum_squared_diff += diff * diff;
    }
    return sqrtf(sum_squared_diff / length);
}

float extract_TD_SKEW(const float32_t* signal, int length) {
    float mean = calculate_mean(signal, length);
    float std = calculate_std(signal, mean, length);

    float sum = 0;
    for(int i = 0; i < length; i++) {
        float z = (signal[i] - mean) / std;
        sum += powf(z, 3);
    }
    return sum / length;
}

float extract_TD_KURT(const float32_t* signal, int length) {
    float mean = calculate_mean(signal, length);
    float std = calculate_std(signal, mean, length);

    float sum = 0;
    for(int i = 0; i < length; i++) {
        float z = (signal[i] - mean) / std;
        sum += powf(z, 4);
    }
    return (sum / length) - 3.0f;
}

void extract_features(const float32_t* signal, int length, float32_t* features) {
    features[ZC_TOTAL_IDX] = extract_ZC_TOTAL(signal, length);
    features[ZC_CHEW_INTERVALS_IDX] = extract_ZC_CHEW_INTERVALS(signal, length);
    features[ZC_NONCHEW_INTERVALS_IDX] = extract_ZC_NONCHEW_INTERVALS(signal, length);
    features[ZC_CHEW_RATIO_IDX] = extract_ZC_CHEW_RATIO(signal, length);
    features[TD_MAX_IDX] = extract_TD_MAX(signal, length);
    features[TD_MIN_IDX] = extract_TD_MIN(signal, length);
    features[TD_SKEW_IDX] = extract_TD_SKEW(signal, length);
    features[TD_KURT_IDX] = extract_TD_KURT(signal, length);
}

static void AI_Init(void) {
	ai_error err;

	const ai_handle act_addr[] = { activations };

	err = ai_svc_linear_one_create_and_init(&proximity_data, act_addr,
			NULL);
	if (err.type != AI_ERROR_NONE) {
		printf("ai_network_create error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
	ai_input = ai_svc_linear_one_inputs_get(proximity_data, NULL);
	ai_output = ai_svc_linear_one_outputs_get(proximity_data, NULL);
}

static void AI_Run(float32_t *pIn, float *pOut) {
	ai_i32 batch;
	ai_error err;

	ai_input[0].data = AI_HANDLE_PTR(pIn);
	ai_output[0].data = AI_HANDLE_PTR(pOut);

	batch = ai_svc_linear_one_run(proximity_data, ai_input, ai_output);
	if (batch != 1) {
		err = ai_svc_linear_one_get_error(proximity_data);
		printf("AI ai_network_run error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
}

int get_frame_activity_class(int frame_start_index, const int* activity_class, int frame_length) {
    int chew_count = 0;

    for (int i = frame_start_index; i < frame_start_index + frame_length; i++) {
        if (i < TEST_DATA_LENGTH && activity_class[i] == 1) {
            chew_count++;
        }
    }

    float chew_ratio = (float)chew_count / frame_length;
    return (chew_ratio >= CHEW_THRESHOLD) ? 1 : 0;
}

void init_metrics(Metrics* metrics) {
    metrics->true_positives = 0;
    metrics->false_positives = 0;
    metrics->false_negatives = 0;
    metrics->true_negatives = 0;
    metrics->precision = 0.0f;
    metrics->recall = 0.0f;
    metrics->f1_score = 0.0f;
    metrics->accuracy = 0.0f;
}

void calculate_metrics(Metrics* metrics) {

    if ((metrics->true_positives + metrics->false_positives) > 0) {
        metrics->precision = (float)metrics->true_positives /
                           (metrics->true_positives + metrics->false_positives);
    }

    if ((metrics->true_positives + metrics->false_negatives) > 0) {
        metrics->recall = (float)metrics->true_positives /
                         (metrics->true_positives + metrics->false_negatives);
    }

    if ((metrics->precision + metrics->recall) > 0) {
        metrics->f1_score = 2.0f * (metrics->precision * metrics->recall) /
                           (metrics->precision + metrics->recall);
    }

    int total = metrics->true_positives + metrics->false_positives +
                metrics->false_negatives + metrics->true_negatives;
    if (total > 0) {
        metrics->accuracy = (float)(metrics->true_positives + metrics->true_negatives) / total;
    }
}

void setup_uart_dma(void) {
    HAL_UART_Receive_DMA(&hlpuart1, dma_buffer, DMA_BUFFER_SIZE);
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    dma_rx_complete = true;
}

void process_dma_data(void) {
    if(dma_rx_complete) {
        float* raw_data = (float*)dma_buffer;

        for(int i = 0; i < NUM_FLOATS; i++) {
            circular_buffer_push(&input_buffer, raw_data[i]);
        }

        dma_rx_complete = false;
        HAL_UART_Receive_DMA(&hlpuart1, dma_buffer, DMA_BUFFER_SIZE);
    }
}

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  PeriphCommonClock_Config();
  MX_GPIO_Init();
  MX_LPUART1_UART_Init();
  
  setup_uart_dma();
  circular_buffer_init(&input_buffer);
  AI_Init();
  Metrics metrics;
  init_metrics(&metrics);

    float32_t data_frame[FRAME_LENGTH];
    float32_t ema_filtered_frame[FRAME_LENGTH];
    float32_t standardized_frame[FRAME_LENGTH];

    float32_t standardized_features[NUM_FEATURES];
    float32_t extracted_features[NUM_FEATURES];
	
    int sample_count = 0;
    int frame_start_idx = 0;
    int track_iter = FIRST_ITERATION;
    float threshold=0.0f;
	
  while (1)
  {
	  for(int sample_idx = 0; sample_idx < TEST_DATA_LENGTH; sample_idx++) {
		  float32_t current_sample=prox_sensor_data[sample_idx];
		  sample_count++;

		  circular_buffer_push(&input_buffer, current_sample);

		  if((track_iter == FIRST_ITERATION) && (sample_count == FRAME_LENGTH)){

			  for (int i = 0; i < FRAME_LENGTH; i++) {
				  circular_buffer_pop(&input_buffer, &data_frame[i]);
			  }

			  for (int i = FRAME_LENGTH / 2; i < FRAME_LENGTH; i++) {
				  circular_buffer_push(&input_buffer, data_frame[i]);
			  }
			  int true_class = get_frame_activity_class(frame_start_idx, activity_class, FRAME_LENGTH);

			  apply_ema_filter(data_frame, ema_filtered_frame, FRAME_LENGTH);

			  standardize_filtered_frame(ema_filtered_frame, standardized_frame, FRAME_LENGTH);

			  extract_features(standardized_frame, FRAME_LENGTH, extracted_features);

			  standardize_features(extracted_features, standardized_features);

			  /* Run inference */
			  AI_Run(standardized_features, aiOutData);

			  int predicted_class = aiOutData[0] > threshold ? 1 : 0;

			  if (predicted_class == 1 && true_class == 1) {
				 metrics.true_positives++;
			 } else if (predicted_class == 1 && true_class == 0) {
				 metrics.false_positives++;
			 } else if (predicted_class == 0 && true_class == 1) {
				 metrics.false_negatives++;
			 } else {
				 metrics.true_negatives++;
			 }

			  track_iter = NOT_FIRST_ITERATION;

			  frame_start_idx += ITER_LENGTH;
		  }

		  if ((sample_count % (int)ITER_LENGTH == 0) && (track_iter == NOT_FIRST_ITERATION) && (sample_count > FRAME_LENGTH)) {

			  for (int i = 0; i < FRAME_LENGTH; i++) {
				  circular_buffer_pop(&input_buffer, &data_frame[i]);
			  }
			  for (int i = FRAME_LENGTH / 2; i < FRAME_LENGTH; i++) {
				  circular_buffer_push(&input_buffer, data_frame[i]);
			  }
			  int true_class = get_frame_activity_class(frame_start_idx, activity_class, FRAME_LENGTH);

			  apply_ema_filter(data_frame, ema_filtered_frame, FRAME_LENGTH);

			  standardize_filtered_frame(ema_filtered_frame, standardized_frame, FRAME_LENGTH);

			  extract_features(standardized_frame, FRAME_LENGTH, extracted_features);

			  standardize_features(extracted_features, standardized_features);

			  /* Run inference */
			  AI_Run(standardized_features, aiOutData);

			  /* Process prediction */
			  int predicted_class = aiOutData[0] > threshold ? 1 : 0;

			  if (predicted_class == 1 && true_class == 1) {
				 metrics.true_positives++;
			 } else if (predicted_class == 1 && true_class == 0) {
				 metrics.false_positives++;
			 } else if (predicted_class == 0 && true_class == 1) {
				 metrics.false_negatives++;
			 } else {
				 metrics.true_negatives++;
			 }

			  frame_start_idx += ITER_LENGTH;
		  }

		  HAL_Delay(5);

	  }
	  calculate_metrics(&metrics);
	  printf("\nFinal Results:\n");
	  printf("True Positives: %d\n", metrics.true_positives);
	  printf("False Positives: %d\n", metrics.false_positives);
	  printf("False Negatives: %d\n", metrics.false_negatives);
	  printf("True Negatives: %d\n", metrics.true_negatives);
	  printf("Accuracy: %.3f\n", metrics.accuracy);
	  printf("Precision: %.3f\n", metrics.precision);
	  printf("Recall: %.3f\n", metrics.recall);
	  printf("F1 Score: %.3f\n", metrics.f1_score);

	  HAL_Delay(20);  
  }
}

