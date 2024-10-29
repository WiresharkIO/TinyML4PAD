To learn and know more about this Tiny-ML project and its general workflow, refer to this documentation: https://wiresharkio.github.io/TinyML4PAD_documentation/

----------------------------
What is this section about ?
---------------------------

Consider a Black-Box where you are getting values from a proximity sensor, as a time-series data. You want to predict and do inference with those values for certain classes using best approaches, such just it is optimized to full possibility and doesn't create any overhead during real-time communications.

You have the sensor and a microcontroller board (in this repository mostly everything is related to STM32-ULP).

---------------------------

> A possible pipeline could be:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/d01518fb-2677-4baa-8f14-3a666e83e03d">


---------------------------

> A more representational pipeline could be:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/99dabd7f-db1e-4a56-957b-356ae59fc311">


---------------------------
Data Collection and Annotation
---------------------------

> One possible approach for data annotation after raw data collection from sensors:

__Label Studio - https://labelstud.io//__

<img width="600" alt="pipeline_annotated_TS_TCD" src="https://github.com/user-attachments/assets/0d8829f1-3bb6-4864-ac3d-3fb70903cbce">


> An example of how configuration can be done in label-studio to represent different classes:
```xml
<View>

<Header value="Time Series classification" style="font-weight: normal"/>
        
<Choices name="pattern" toName="ts">
    <Choice value="Left_Prox"/>
</Choices>

<TimeSeriesLabels name="label" toName="ts">
    <Label value="phase_1" background="red"/>
    <Label value="phase_2" background="green"/>
    <Label value="phase_3" background="yellow"/>
    <Label value="phase_4" background="blue"/>
</TimeSeriesLabels>

<TimeSeries name="ts" value="$Left_Prox" valueType="url">
    <Channel column="Left_Prox"/>
</TimeSeries>
   
</View>
```

---------------------------
__How the data could look like:__

<img width="600" alt="annotated_TS_TCD" src="https://github.com/user-attachments/assets/e7648c5d-e48f-4f69-a6f7-86090d56dac6">


---------------------------
# Feature Engineering 
---------------------------

__Transforming raw data into features__

__The performance of ML models heavily depends on the relevance of the features used to train them.__

5 processes in feature engineering:
-->Feature Creation
-->Feature Transformation
-->Feature Extraction
-->Feature Selection
-->Feature Scaling

<img width="600" src="https://github.com/user-attachments/assets/471ed7df-816f-4eca-a1d7-9850012cb563">

Ref: https://www.geeksforgeeks.org/what-is-feature-engineering/


Some of the feature selection methods used here, in-general and in-specific to LinearSVC(the model)


> In general approach :

1. minimum Redundancy - Maximum Relevance(mRMR) - https://github.com/smazzanti/mrmr

2. SHAP(SHapley Additive exPlanations) - explains the impact of different features on the output of the model. Its values show how much each 
feature contributes to the model's decision, providing insight into feature importance.
<img width="600" alt="image" src="https://github.com/user-attachments/assets/3e2300ad-b6dd-47e1-b41a-43cdceac1805">

Features at the top have a higher impact on the model’s output(importance) compared to those at the bottom. The more the spread from 0, 
the larger the impact of that particular feature.


> Model specific approach :

__SVM weighting__ - using the absolute coefficients. These weights/coefficients tells how important each feature is to the decision boundary 
created by the model. The larger the absolute value of the weight, the more significant the feature is for classification.

__Recursive feature elimination__ - least important features are pruned from the current set of features.

| see example |
```python
svc = LinearSVC(C=0.1, dual=False, loss='squared_hinge', penalty='l2', class_weight={0: 1, 1: 10}, max_iter=10000, verbose=True)
rfe = RFE(estimator=svc, n_features_to_select=30, step=1)
rfe.fit(X_train_standardized, y_train)
X_train_rfe = rfe.transform(X_train_standardized)
X_test_rfe = rfe.transform(X_test_standardized)
svc.fit(X_train_rfe, y_train)
```

> Performance plot of Teacher-Student knowledge distillation model :

<img width="600" alt="student_model_performance" src="https://github.com/user-attachments/assets/f45acf00-0f73-484c-81c0-b636632edf59">


TO DO..

- ~~Recursive Feature Elimination (RFE).~~
- Improve model metrics of LinearSVC.
- Implement EfficientNet or EfficientNet-Lite
- Explore On-device learning
- Explore federated learning
---------------------------



