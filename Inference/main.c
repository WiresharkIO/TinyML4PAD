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

#include "svm_linear_lopo1.h"
#include "svm_linear_lopo1_data.h"
#include "svm_linear_lopo1_data_params.h"

#include "ai_datatypes_defines.h"
#include "ai_platform.h"
#define TOTAL_SAMPLES_PER_SEGMENT 40
#define NUM_SEGMENTS 2

#include "test_data_LOPO_01.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* for inference performance */
typedef struct {
	ai_float features[NUM_FEATURES];
    uint8_t true_label;
} TestSegment;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

// FOR DATA TRANSMISSON TO SERIAL TERMINAL
int8_t strTemp[100];
char strPrediction[100];

// AI-MODEL required buffers/variables
ai_handle proximity_data;
AI_ALIGNED(4) ai_float aiInData[AI_SVM_LINEAR_LOPO1_IN_1_SIZE];
static AI_ALIGNED(4) ai_float aiOutData[AI_SVM_LINEAR_LOPO1_OUT_1_SIZE];
ai_u8 activations[AI_SVM_LINEAR_LOPO1_DATA_ACTIVATIONS_SIZE];
ai_buffer *ai_input;
ai_buffer *ai_output;

// FOR INFERENCE PREDICTION
const char *activities[2] = { "NotEating", "Eating" };
AI_ALIGNED(4) ai_float model_output;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
static void MX_GPIO_Init(void);
/* USER CODE BEGIN PFP */
static void AI_Init(void);
static void AI_Run(float *pIn, float *pOut);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// TEST-INPUT
TestSegment test_segments[] = {
    {{0.096506460000000, -0.164920820000000, 0.135400350000000, 0.184841160000000,
    -0.070261580000000, -0.030177400000000, 0.187843610000000, -0.093096590000000,
	-1.263773850000000, 0.359632190000000, -0.720365040000000, 0.225761960000000,
	-0.739728250000000, -0.461497390000000, 1.250458810000000, 0.015801680000000,
	0.836598060000000, -1.227048390000000, 1.819182770000000, 1.814093060000000,
	1.403826530000000, -0.828473260000000, 2.549969770000000, 1.972066940000000,
	0.577904340000000, -0.449631810000000, 1.089196650000000, 0.577904340000000,
	1.104573730000000, 1.422227970000000, 0.999091130000000, 1.360588130000000,
	1.810035400000000, 0.653906910000000, 0.653906910000000, -0.309108940000000,
	-0.309108940000000, -0.313859680000000, -0.336692320000000, 2.568411550000000}, 1},

	{{-0.066060780000000, 0.034536410000000, -0.052621520000000, -0.058222210000000,
	0.011438600000000, -0.032548970000000, -0.057762590000000, -3.910138820000000,
	5.843828230000000, -0.076920650000000, -0.955262630000000, -1.080793430000000,
	-1.303231160000000, -0.783729820000000, -0.910725590000000, -1.132649230000000,
	-0.962573720000000, 0.030126640000000, -0.160119340000000, -0.212351720000000,
	0.031741030000000, -1.050244510000000, 0.046151890000000, 0.282148890000000,
	-0.847784760000000, -0.449631810000000, -0.740239320000000, -0.847784760000000,
	-0.465029830000000, -0.368028120000000, -0.509093100000000, -0.363048430000000,
	-0.250355450000000, -0.298545390000000, -0.298545390000000, -0.311823820000000,
	-0.311823820000000, -0.009272620000000, -0.308397660000000, 0.369353730000000}, 0}
};

static void AI_Init(void) {
	ai_error err;

	const ai_handle act_addr[] = { activations };

	err = ai_svm_linear_lopo1_create_and_init(&proximity_data, act_addr,
			NULL);
	if (err.type != AI_ERROR_NONE) {
		printf("ai_network_create error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
	ai_input = ai_svm_linear_lopo1_inputs_get(proximity_data, NULL);
	ai_output = ai_svm_linear_lopo1_outputs_get(proximity_data, NULL);
}

static void AI_Run(float *pIn, float *pOut) {
	ai_i32 batch;
	ai_error err;

	ai_input[0].data = AI_HANDLE_PTR(pIn);
	ai_output[0].data = AI_HANDLE_PTR(pOut);

	batch = ai_svm_linear_lopo1_run(proximity_data, ai_input, ai_output);
	if (batch != 1) {
		err = ai_svm_linear_lopo1_get_error(proximity_data);
		printf("AI ai_network_run error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
}

uint8_t make_prediction(ai_float model_output) {
	return (model_output > 0) ? 1 : 0;
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

  /* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USB_Device_Init();
  /* USER CODE BEGIN 2 */
	DWT_Init();
	AI_Init();

	uint32_t correct_predictions = 0;
	uint32_t total_predictions = 0;

	uint32_t start_cycle, end_cycle;
	ai_float cpu_mhz = SystemCoreClock / 1000000.0f;
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {

	  /* ADDED TO INFER 2-INPUT SEGEMENTS HARD-CODED IN THIS FILE */
	  /*
	  for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
			memcpy(aiInData, test_segments[seg].features, sizeof(ai_float) * TOTAL_SAMPLES_PER_SEGMENT);
			start_cycle = DWT_GetCycle();
			AI_Run(aiInData, aiOutData);
			end_cycle = DWT_GetCycle();
			ai_float inference_time = (end_cycle - start_cycle) / (cpu_mhz * 1000.0f);

			sprintf(strTemp, "Inference time: %0.3f msec\r\n", inference_time);
			CDC_Transmit_FS((uint8_t*) strTemp, strlen(strTemp));
			HAL_Delay(10);

			uint8_t predicted_class = make_prediction(aiOutData[0]);

			if (predicted_class == test_segments[seg].true_label) {
				correct_predictions++;
			}
			total_predictions+=1;

			sprintf(strPrediction, "Segment-%lu, Predicted_class-%d, True_class-%d\r\n",
								seg, predicted_class, test_segments[seg].true_label);
			CDC_Transmit_FS((uint8_t*) strPrediction, strlen(strPrediction));
			HAL_Delay(10);
	  	  }
	  float accuracy = (float)correct_predictions / total_predictions * 100.0f;
	  sprintf(strTemp, "Accuracy: %.2f%% \r\n", accuracy);
	  CDC_Transmit_FS((uint8_t*) strTemp, strlen(strTemp));
	  HAL_Delay(10);
	  */

	  /* ADDED TO INFER BLOCK OF INPUT */

	  for (uint32_t segment = 0; segment < NUM_SAMPLES; segment++) {
	      memcpy(aiInData, sampled_data[segment], sizeof(ai_float) * NUM_FEATURES);

	      start_cycle = DWT_GetCycle();
	      AI_Run(aiInData, aiOutData);
	      end_cycle = DWT_GetCycle();

	      ai_float inference_time = (end_cycle - start_cycle) / (cpu_mhz * 1000.0f);

	      uint8_t predicted_class = make_prediction(aiOutData[0]);
	      uint8_t true_label = (int)sampled_data[segment][NUM_FEATURES];
	      if (predicted_class == true_label) {
	    	  correct_predictions++;
	      }
	      total_predictions+=1;

	      sprintf(strTemp, "Segment %lu, Inference time: %.3f msec\r\n", segment, inference_time);
	      CDC_Transmit_FS((uint8_t*) strTemp, strlen(strTemp));
	      HAL_Delay(10);

	  }

	  float accuracy = (float)correct_predictions / total_predictions * 100.0f;
	  sprintf(strTemp, "Accuracy: %.2f%% \r\n", accuracy);
	  CDC_Transmit_FS((uint8_t*) strTemp, strlen(strTemp));
	  HAL_Delay(10);
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
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
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_LSE
                              |RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.LSEState = RCC_LSE_OFF;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSICalibrationValue = RCC_MSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV1;
  RCC_OscInitStruct.PLL.PLLN = 32;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the SYSCLKSource, HCLK, PCLK1 and PCLK2 clocks dividers
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK4|RCC_CLOCKTYPE_HCLK2
                              |RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.AHBCLK2Divider = RCC_SYSCLK_DIV2;
  RCC_ClkInitStruct.AHBCLK4Divider = RCC_SYSCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enable MSI Auto calibration
  */
  HAL_RCCEx_EnableMSIPLLMode();
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_SMPS;
  PeriphClkInitStruct.SmpsClockSelection = RCC_SMPSCLKSOURCE_HSI;
  PeriphClkInitStruct.SmpsDivSelection = RCC_SMPSCLKDIV_RANGE0;

  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN Smps */

  /* USER CODE END Smps */
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
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_5, GPIO_PIN_RESET);

  /*Configure GPIO pin : PC4 */
  GPIO_InitStruct.Pin = GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : PB0 PB1 PB5 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : PD0 PD1 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : PB6 PB7 */
  GPIO_InitStruct.Pin = GPIO_PIN_6|GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART1;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

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
