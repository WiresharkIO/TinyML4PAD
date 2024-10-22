/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
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
#define OVERLAP 0.5f /* the amount of samples overlapping with the next segment, example
0-->150(segment-1), 75-->225(segment-2), ..so on */
#define ITER_LENGTH (FRAME_LENGTH * (1.0f - OVERLAP)) /* 150-samples*(1-0.5)=75-samples  */
#define FFT_SIZE 256  // Next power of 2 after 150
/* Butterworth-BPF */
#define BLOCK_SIZE 256
#define NUM_TAPS 11 // 5th order filter = 5 * 2 + 1 taps
#define NUM_STAGES 3 // 5th order filter = 3 biquad stages

#define FEATURE_COUNT 7 // Number of features we're extracting

/* validation parameters */
#define TEST_DATA_SIZE 1500


/* data-gathering and storing in circular buffer to extract features for inference */
typedef struct {
    float data[FRAME_LENGTH];
    int head;
    int tail;
    int count;
} CircularBuffer;
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

/* signal processing parameters for feature extraction in raw data(signal) */
//float filtered_data[DATA_POINTS];
//float extracted_features[7];

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
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* data-gathering and storing in circular buffer to extract features for inference */
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
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
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

    // Normalize coefficients
    b0 /= a0;
    b1 /= a0;
    b2 /= a0;
    a1 /= a0;
    a2 /= a0;

    // Set up coefficient array for biquad filter
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
    // Create a copy of the data so the original array is not modified
    float sorted_data[length];
    for (int i = 0; i < length; i++) {
        sorted_data[i] = data[i];
    }

    // Step 1: Sort the copy of the array
    qsort(sorted_data, length, sizeof(float), compare_floats);

    // Step 2: Calculate the median
    if (length % 2 == 0) {
        // Even number of elements: median is the average of the two middle elements
        *median = (sorted_data[length/2 - 1] + sorted_data[length/2]) / 2.0f;
    } else {
        // Odd number of elements: median is the middle element
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
//	const int FFT_SIZE = 256;  // Next power of 2 after 150
	// Create temporary buffers for FFT computation
	static float fft_buffer[2 * FFT_SIZE]; // Complex input buffer (twice the size for real/imaginary pairs)
	static float fourier_frame[FFT_SIZE];
	static float psd_frame[FFT_SIZE/2];
//    float fourier_frame[FRAME_LENGTH];
//    float psd_frame[FRAME_LENGTH/2];
//    int fft_length=FRAME_LENGTH / 2;
	// Clear the buffers
	memset(fft_buffer, 0, sizeof(fft_buffer));
	memset(fourier_frame, 0, sizeof(fourier_frame));
	memset(psd_frame, 0, sizeof(psd_frame));


	for (int i = 0; i < FRAME_LENGTH; i++) {
		fft_buffer[2*i] = window_frame[i];    // Real part
		fft_buffer[2*i + 1] = 0.0f;           // Imaginary part
	}

    // Compute FFT
//    arm_cfft_radix4_instance_f32 S;
//    arm_cfft_radix4_init_f32(&S, fft_length, 0, 1);
//    arm_cfft_radix4_f32(&S, window_frame);
//    arm_cmplx_mag_f32(window_frame, fourier_frame, fft_length);

	arm_cfft_instance_f32 S;
	arm_status status = arm_cfft_init_f32(&S, FFT_SIZE);
	if (status != ARM_MATH_SUCCESS) {
		// Handle initialization error
		flag=999;
		return;
	}

	arm_cfft_f32(&S, fft_buffer, 0, 1);
	// Calculate magnitude
	for (int i = 0; i < FFT_SIZE/2; i++) {
		float real = fft_buffer[2*i];
		float imag = fft_buffer[2*i + 1];
		fourier_frame[i] = sqrtf(real * real + imag * imag);
	}

    // Compute PSD
    float winq = 0;
    arm_power_f32(hann_window, FRAME_LENGTH, &winq);
//    winq *= 25;
    winq *= FS;
//    arm_mult_f32(fourier_frame, fourier_frame, psd_frame, fft_length);
//    arm_scale_f32(psd_frame, 1/winq, psd_frame, fft_length);
    for (int i = 0; i < FFT_SIZE/2; i++) {
		psd_frame[i] = (fourier_frame[i] * fourier_frame[i]) / winq;
	}
    // Compute features
    features[0] = calculate_kurtosis(frame, FRAME_LENGTH);  // TD_KURT
//    arm_median_f32(fourier_frame, FRAME_LENGTH/2, &features[1]);
//    calculate_median(fourier_frame, fft_length, &features[1]); // FD_MEDIAN
//    arm_max_f32(psd_frame, fft_length, &features[2], NULL);  // TFD_MAX
//    arm_std_f32(psd_frame, fft_length, &features[3]);  // TFD_STD
//    features[4] = calculate_entropy(psd_frame, fft_length);  // TFD_S_ENT
//    features[5] = calculate_kurtosis(psd_frame, fft_length);  // TFD_S_KURT
//    features[6] = calculate_kurtosis(psd_frame, fft_length);  // TFD_KURT
//    features[1]=2.0f;
//    features[2]=2.0f;
//    features[3]=2.0f;
//    features[4]=2.0f;
//    features[5]=2.0f;
//    features[6]=2.0f;
    calculate_median(fourier_frame, FFT_SIZE/2, &features[1]); // FD_MEDIAN
	arm_max_f32(psd_frame, FFT_SIZE/2, &features[2], NULL);  // TFD_MAX
	arm_std_f32(psd_frame, FFT_SIZE/2, &features[3]);  // TFD_STD
	features[4] = calculate_entropy(psd_frame, FFT_SIZE/2);  // TFD_S_ENT
	features[5] = calculate_kurtosis(psd_frame, FFT_SIZE/2);  // TFD_S_KURT
	features[6] = calculate_kurtosis(psd_frame, FFT_SIZE/2);  // TFD_KURT
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

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_USB_DEVICE_Init();
  /* USER CODE BEGIN 2 */
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
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
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

			  // Perform inference
			  start_cycle = DWT_GetCycle();
			  AI_Run(standardized_features, aiOutData);
			  end_cycle = DWT_GetCycle();

			  // Get the model's prediction
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
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  return 0;
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 72;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 3;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
