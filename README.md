---------------------
__Tiny Machine Learning for Proximity Sensor Activity Detection__
---------------------

> Design Cycle
<img width="600" alt="image" src="https://github.com/user-attachments/assets/0e791e2f-f819-4daf-8542-740fbb47e352">


---------------------
> Problems existing

<img width="200" alt="image" src="https://github.com/user-attachments/assets/fecb3756-a139-49f5-bcd5-d05fec972a44">

--> Yet to get a new one..


> Problems resolved

1. "Not able to find Index" in STM32 Cube IDE during Building project
- Try with Project--> c/c++ Index --> Rebuild

2. Not able to connect to serial terminal like "putty"
- Try with checking:
  1. windows-->device manager--> desired COM port properties and set desired communication rate and other vals.
  2. check/set proper TIMER parameters in IDE configuration window.
  3. if using putty check serial communiation configuration and set it equal to the values set elsewhere.

3. During quantization - InferenceError: [ShapeInferenceError] (op_type:ZipMap, node name: ZipMap): [ShapeInferenceError] type case unsupported for symbolic shape inference. inferred=5
- Refer --> https://github.com/onnx/sklearn-onnx/issues/816

4. Invalid Initializer error with auto-generated code X-CUBE-AI
- Refer --> https://community.st.com/t5/edge-ai/invalid-initializer-error-with-auto-generated-code-x-cube-ai/td-p/104534

5. [STM32F411-Discovery Board] HAL_RCC_OscConfig() always returns HSE_TIMEOUT
- Changing from BYPASS clock source to Crystal/Ceramic Resonator in RCC and then power-cycling(RESET) the board was the solution.

6. https://community.st.com/t5/stm32-mcus-boards-and-hardware/i-could-not-upgrade-st-link-firmware-i-always-get-an-error-using/td-p/220944

---------------------
> __References__

[1] Tsoukas, V. et al. (2024) ‘A review on the emerging technology of tinyml’, ACM Computing Surveys, 56(10), pp. 1–37. doi:10.1145/3661820. 

[2] "Model Optimization", ONNX Runtime, Available: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
