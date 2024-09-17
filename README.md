# Predicting Student Dropout with Supervised Learning

## Project Overview
This project utilises supervised learning models to predict student dropout rates using a dataset from an anonymised educational services provider. The analysis aims to identify key factors influencing student dropout, providing insights to enhance student retention.

## Problem Statement
High dropout rates can have a significant impact on educational institutions, leading to revenue loss, diminished reputation, and lower overall satisfaction. The objective of this analysis is to predict which students are most likely to drop out based on various factors, allowing for targeted interventions to improve retention rates.

## Methodology
1. **Data Exploration:** Initial exploration to identify key features, followed by data cleaning, feature engineering, and preprocessing.
2. **Feature Selection:** Two datasets were prepared:
   - **X1 Dataset:** Original key features.
   - **X2 Dataset:** Original features plus 'AttendancePercentage' and 'ContactHours'.
3. **Modeling Approaches:** 
   - **XGBoost:** Both default parameters and hyperparameter-tuned models were used.
   - **Neural Network:** Implemented using TensorFlow/Keras with custom metrics for evaluating performance.
4. **Hyperparameter Tuning:** Utilised grid search with the F1 score as the evaluation metric to address dataset imbalance.

## Key Insights
- **CreditWeightedAverage:** Identified as the most important feature in predicting student dropout.
- **Centre Name:** Location plays a significant role, indicating the impact of different resources, support systems, and demographics across centres.
- Adding 'AttendancePercentage' and 'ContactHours' notably improved model performance, highlighting their importance in predicting dropout rates.

## Techniques and Libraries
- **Supervised Learning Models:** XGBoost and Neural Network (TensorFlow/Keras).
- **Evaluation Metrics:** F1 score, accuracy, precision, recall, and AUC.
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `tensorflow`, `keras-tuner`.

## Why Focus on F1 Score Over Accuracy?
The dataset used in this project is highly imbalanced, with significantly more students completing their courses than dropping out. In such cases, accuracy can be misleading, as a model predicting all students as "not dropping out" would still yield high accuracy due to the majority class. 

The **F1 score** was chosen as the key evaluation metric because it provides a better balance between precision and recall. This metric is particularly useful when the cost of false positives (predicting a student will drop out when they won't) and false negatives (missing an actual dropout) is high. By focusing on the F1 score, the model is optimised to correctly identify students at risk of dropping out without being overly influenced by the majority class.

## Parallels to Customer Churn in Other Sectors
Predicting student dropout is similar to customer churn in sectors e.g. retail. Both involve analysing historical behaviours (e.g., attendance vs. usage frequency) to identify at-risk individuals. Recommendations derived here can be adapted to retention strategies in various industries.

## Data Overview
The dataset contains 25,060 rows representing student records. Key features include:
- **Age at Arrival**
- **Credit Weighted Average**
- **Attendance Percentage**
- **Contact Hours**
- **Unauthorised Absence Count**
- **Centre Name** and others.

## Findings
- **CreditWeightedAverage:** Lower scores significantly increase dropout likelihood.
- **Centre Impact:** Some centres exhibit higher dropout rates, suggesting a need for tailored support.
- **Model Performance:** 
  - **XGBoost:** Outperformed the neural network, achieving a maximum F1-score of 0.85 for predicting dropouts.
  - **Neural Network:** Performed comparably, with an F1-score of 0.81, though it showed slight sensitivity to added features.

### Recommendations
1. **Enhance Academic Support:** Focus on improving students' academic performance through tutoring services and study groups.
2. **Strengthen Centre-Specific Support:** Address unique challenges in centres with higher dropout rates.
3. **Monitor Attendance:** Develop initiatives that promote regular attendance to mitigate dropout risk.

## Files in This Repository
- **Predicting_Student_Dropout_with_Supervised_Learning.ipynb:** The Jupyter Notebook containing the full analysis, including data preprocessing, model implementation, and visualisation.
- **LICENSE:** The license file for the project.
- *Note:* The raw data has not been included to comply with data privacy standards.

## Usage
Open the `Predicting_Student_Dropout_with_Supervised_Learning.ipynb` notebook to explore the full analysis, which includes data processing, model building, and insights on student dropout prediction.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
