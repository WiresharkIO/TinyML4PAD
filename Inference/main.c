#include "main.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "test_data.h"

/* math operation includes */
#include <math.h>
#include "arm_math.h"
//#include "arm_const_structs.h"
//#include "fatfs.h"

/* model file includes */
#include "svclinear.h"
#include "svclinear_data.h"
#include "svclinear_data_params.h"

#include "ai_datatypes_defines.h"
#include "ai_platform.h"

/* signal processing parameters for feature extraction in raw data(signal) */
#define FS 50 /* sampling frequency i.e the number of samples per second */
#define SEGMENT_LENGTH 3 /* in seconds */
#define FRAME_LENGTH (FS * SEGMENT_LENGTH) /* total number of samples per segment */
//#define FRAME_LENGTH 128

/* the amount of samples overlapping with the next segment, example
0-->150(segment-1), 75-->225(segment-2), ..so on */
#define OVERLAP 0.5f 
/* 150-samples*(1-0.5)=75-samples */
#define ITER_LENGTH (FRAME_LENGTH * (1.0f - OVERLAP)) 
/* Next power of 2 after 150 - to keep it in powers of 2 */
#define FFT_SIZE 256  

/* Butterworth-BPF */
#define BLOCK_SIZE 256
#define NUM_TAPS 11 // 5th order filter = 5 * 2 + 1 taps
#define NUM_STAGES 3 // 5th order filter = 3 biquad stages

#define FEATURE_COUNT 7 // Number of features we're extracting..

/* validation parameters */
#define TEST_DATA_SIZE 1500


/* data-gathering and storing in circular buffer to extract features for inference */
typedef struct {
    float data[FRAME_LENGTH];
    int head;
    int tail;
    int count;
} CircularBuffer;

/* hanning-window */
float hann_window[FRAME_LENGTH];

/* Butterworth-BPF filter coefficients */
//float32_t b[] = {
//    0.00081629, 0.0, -0.00408144, 0.0, 0.00816288, 0.0,
//    -0.00816288, 0.0, 0.00408144, 0.0, -0.00081629
//};
//float32_t a[] = {
//    1.0, -8.01123129, 29.06510762, -62.94704732, 90.18674136,
//    -89.36422836, 62.03718586, -29.79614922, 9.47586324, -1.80177451,
//    0.15553267
//};
arm_biquad_casd_df1_inst_f32 S;
/* state buffer to store states of BPF */
static float32_t state[4 * NUM_STAGES];
/* coeff. buffer to store coeffs. of BPF */
static float32_t coeff[5 * NUM_STAGES];

/* data-gathering and storing in circular buffer to extract features for inference */
CircularBuffer input_buffer;

/* Training data standardization values for feature */
const float feature_means[FEATURE_COUNT] = {0.40782377f, 0.00054899f, 0.59396853f, 0.09704273f, 2.10049263f, 19.23477190f, 17.29472961f};
const float feature_stds[FEATURE_COUNT] = {1.20171863f, 0.00034129f, 0.40976195f, 0.05771965f, 0.65923357f, 11.58719020f, 11.47957269f};
int flag=888;

/* model IO variables and buffers */
//float sensor_data_buffer[DATA_POINTS];
ai_handle proximity_data;
AI_ALIGNED(4) ai_float aiInData[AI_SVCLINEAR_IN_1_SIZE];
static AI_ALIGNED(4) ai_float aiOutData[AI_SVCLINEAR_OUT_1_SIZE];
ai_u8 activations[AI_SVCLINEAR_DATA_ACTIVATIONS_SIZE];
ai_buffer *ai_input;
ai_buffer *ai_output;

/* FOR INFERENCE PREDICTION */
const char *activities[2] = { "NotEating", "Eating" };
AI_ALIGNED(4) ai_float model_output;

void circular_buffer_init(CircularBuffer* cb);
void circular_buffer_push(CircularBuffer* cb, float item);
void circular_buffer_pop(CircularBuffer* cb, float* item);

/* Butterworth-BPF */
void bandpass_filter_init(float f_low, float f_high, float fs);
void apply_bandpass_filter(float* input, float* output, int length);

/* hanning-window */
void arm_hanning_f32_init(float* frame, int length);

/* feature extraction function prototypes */
int compare_floats(const void* a, const void* b);
void calculate_median(float* data, int length, float* median);
float calculate_kurtosis(float* data, int length);
float calculate_entropy(float* data, int length);
void extract_features(float* frame, float* window_frame, float* features);

/* model related function prototypes */
//void receive_sensor_data(float new_data_point);
static void AI_Init(void);
static void AI_Run(float *pIn, float *pOut);

void DWT_Init(void);
uint32_t DWT_GetCycle(void);

static void AI_Init(void) {
	ai_error err;

	const ai_handle act_addr[] = { activations };

	err = ai_svclinear_create_and_init(&proximity_data, act_addr,
			NULL);
	if (err.type != AI_ERROR_NONE) {
		printf("ai_network_create error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
	ai_input = ai_svclinear_inputs_get(proximity_data, NULL);
	ai_output = ai_svclinear_outputs_get(proximity_data, NULL);
}

static void AI_Run(float *pIn, float *pOut) {
	ai_i32 batch;
	ai_error err;

	ai_input[0].data = AI_HANDLE_PTR(pIn);
	ai_output[0].data = AI_HANDLE_PTR(pOut);

	batch = ai_svclinear_run(proximity_data, ai_input, ai_output);
	if (batch != 1) {
		err = ai_svclinear_get_error(proximity_data);
		printf("AI ai_network_run error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
}

/* storing and restoring data in a circular buffer */
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

/* Butterworth-BPF */
void bandpass_filter_init(float f_low, float f_high, float fs) {
    float w_low = 2.0f * tanf(M_PI * f_low / fs);
    float w_high = 2.0f * tanf(M_PI * f_high / fs);
    float w0 = sqrtf(w_low * w_high);
    float bw = w_high - w_low;
    float Q = w0 / bw;
    float alpha = sinf(w0) / (2.0f * Q);

    float b0 = alpha;
    float b1 = 0.0f;
    float b2 = -alpha;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cosf(w0);
    float a2 = 1.0f - alpha;

    b0 /= a0;
    b1 /= a0;
    b2 /= a0;
    a1 /= a0;
    a2 /= a0;

    for (int i = 0; i < NUM_STAGES; i++) {
        coeff[5*i] = b0;
        coeff[5*i + 1] = b1;
        coeff[5*i + 2] = b2;
        coeff[5*i + 3] = -a1;
        coeff[5*i + 4] = -a2;
    }

    // Initialize the filter
    arm_biquad_cascade_df1_init_f32(&S, NUM_STAGES, coeff, state);
}

void apply_bandpass_filter(float* input, float* output, int length) {
    arm_biquad_cascade_df1_f32(&S, input, output, length);
}

/* hanning-window implementation */
void arm_hanning_f32_init(float *hann_window, int frame_length) {
    for (uint32_t i = 0; i < frame_length; i++) {
    	hann_window[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (frame_length - 1)));
    }
}

float calculate_kurtosis(float* data, int length) {
    float mean, var, kurt;
    arm_mean_f32(data, length, &mean);
    arm_var_f32(data, length, &var);

    float sum = 0;
    for (int i = 0; i < length; i++) {
        float diff = data[i] - mean;
        sum += diff * diff * diff * diff;
    }

    kurt = (sum / length) / (var * var) - 3;
    return kurt;
}

int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);  // Returns 1 if fa > fb, -1 if fa < fb, 0 otherwise
}

void calculate_median(float* data, int length, float* median) {
    float sorted_data[length];
    for (int i = 0; i < length; i++) {
        sorted_data[i] = data[i];
    }
	
    qsort(sorted_data, length, sizeof(float), compare_floats);

    if (length % 2 == 0) {
        *median = (sorted_data[length/2 - 1] + sorted_data[length/2]) / 2.0f;
    } else {
        *median = sorted_data[length/2];
    }
}

float calculate_entropy(float* data, int length) {
    float sum = 0;
    for (int i = 0; i < length; i++) {
        if (data[i] > 0) {
            sum += data[i] * log2f(data[i]);
        }
    }
    return -sum;
}

void extract_features(float* frame, float* window_frame, float* features) {

	static float fft_buffer[2 * FFT_SIZE];
	static float fourier_frame[FFT_SIZE];
	static float psd_frame[FFT_SIZE/2];
	memset(fft_buffer, 0, sizeof(fft_buffer));
	memset(fourier_frame, 0, sizeof(fourier_frame));
	memset(psd_frame, 0, sizeof(psd_frame));

	for (int i = 0; i < FRAME_LENGTH; i++) {
		fft_buffer[2*i] = window_frame[i];    // Real part
		fft_buffer[2*i + 1] = 0.0f;           // Imaginary part
	}

	arm_cfft_instance_f32 S;
	arm_status status = arm_cfft_init_f32(&S, FFT_SIZE);
	if (status != ARM_MATH_SUCCESS) {
		// Handle initialization error
		flag=999;
		return;
	}

	arm_cfft_f32(&S, fft_buffer, 0, 1);
	for (int i = 0; i < FFT_SIZE/2; i++) {
		float real = fft_buffer[2*i];
		float imag = fft_buffer[2*i + 1];
		fourier_frame[i] = sqrtf(real * real + imag * imag);
	}
	
    float winq = 0;
    arm_power_f32(hann_window, FRAME_LENGTH, &winq);
//    winq *= 25;
    winq *= FS;
    for (int i = 0; i < FFT_SIZE/2; i++) {
		psd_frame[i] = (fourier_frame[i] * fourier_frame[i]) / winq;
	}

    features[0] = calculate_kurtosis(frame, FRAME_LENGTH);
//    features[1]=2.0f;
//    features[2]=2.0f;
//    features[3]=2.0f;
//    features[4]=2.0f;
//    features[5]=2.0f;
//    features[6]=2.0f;
    calculate_median(fourier_frame, FFT_SIZE/2, &features[1]);
	arm_max_f32(psd_frame, FFT_SIZE/2, &features[2], NULL);
	arm_std_f32(psd_frame, FFT_SIZE/2, &features[3]);
	features[4] = calculate_entropy(psd_frame, FFT_SIZE/2);
	features[5] = calculate_kurtosis(psd_frame, FFT_SIZE/2);
	features[6] = calculate_kurtosis(psd_frame, FFT_SIZE/2);
}

void standardize_features(float* features, float* standardized_features) {
    for (int i = 0; i < FEATURE_COUNT; i++) {
        standardized_features[i] = (features[i] - feature_means[i]) / feature_stds[i];
    }
}

void DWT_Init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t DWT_GetCycle(void) {
    return DWT->CYCCNT;
}

int main(void)
{
  float f_low = 0.5f;
  float f_high = 5.0f;

  bandpass_filter_init(f_low, f_high, FS);
  arm_hanning_f32_init(&hann_window, FRAME_LENGTH);
  circular_buffer_init(&input_buffer);

  float frame[FRAME_LENGTH];
  float window_frame[FRAME_LENGTH];
  float normalized_frame[FRAME_LENGTH];
  float extracted_features[FEATURE_COUNT];
  float mean_value, std_value;
  int sample_count = 0;
  float standardized_features[FEATURE_COUNT];

  AI_Init();
  DWT_Init();
  
  uint32_t start_cycle, end_cycle;
  int correct_predictions = 0;
  int total_predictions = 0;
  
  while (1)
  {
	  for (int i = 0; i < TEST_DATA_SIZE; i++){
		  float new_sample=raw_input_data[i];

		  circular_buffer_push(&input_buffer, new_sample);
		  sample_count++;

		  if (sample_count % (int)ITER_LENGTH == 0) {
			  for (int i = 0; i < FRAME_LENGTH; i++) {
				  circular_buffer_pop(&input_buffer, &frame[i]);
			  }
			  for (int i = FRAME_LENGTH / 2; i < FRAME_LENGTH; i++) {
				  circular_buffer_push(&input_buffer, frame[i]);
			  }

			  apply_bandpass_filter(frame, frame, FRAME_LENGTH);
	//		  arm_mult_f32(frame, hann_window, frame, FRAME_LENGTH);
			  arm_mean_f32(frame, FRAME_LENGTH, &mean_value);
			  arm_std_f32(frame, FRAME_LENGTH, &std_value);

			  if (std_value != 0) {
				  for (int i = 0; i < FRAME_LENGTH; i++) {
					  normalized_frame[i] = (frame[i] - mean_value) / std_value;
				  }
			  }

			  arm_mult_f32(normalized_frame, hann_window, window_frame, FRAME_LENGTH);
			  extract_features(frame, window_frame, extracted_features);
			  standardize_features(extracted_features, standardized_features);
			  
			  start_cycle = DWT_GetCycle();
			  AI_Run(standardized_features, aiOutData);
			  end_cycle = DWT_GetCycle();

			  /* prediction */
			  float prediction = aiOutData[0];
			  int predicted_class = prediction > 0.5 ? 1 : 0;  // Assuming binary classification
	//		  infer_model(features);
			  if (predicted_class == activity_class[i]) {
				  correct_predictions++;
			  }
			  total_predictions++;
		  }
	  }
	  float accuracy = (float)correct_predictions / total_predictions;
	  printf("Accuracy: %.2f%%\n", accuracy * 100);
	  HAL_Delay(20);
  }
  return 0;
}