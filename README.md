# End to End Machine Learning Project

# AMEX Credit Default Prediction

## Project Overview

The AMEX Credit Default Prediction project aims to develop a robust machine learning pipeline for predicting credit default risk among customers. The model will analyze customer transaction data and associated features to determine the likelihood of default. This project is structured to facilitate data processing, model training, prediction generation, and result evaluation.


## Features

- **Data Loading:**
  - Efficiently loads large datasets in Feather format for optimal performance.
  - Supports both training and test datasets.

- **Data Preprocessing:**
  - Cleans and transforms the data to ensure it is suitable for model training.
  - Imputating missing values with KNN algorithms, encodes categorical variables.

- **Model Training:**
  - Implements a variety of machine learning algorithms, including Random Forest and LGBM.
  - Conducts hyperparameter tuning using cross-validation to optimize model performance.
  - Tracks model performance metrics such as roc auc score and gini metrics.

- **Prediction Generation:**
  - Generates predictions on test data using the trained model.

- **Result Storage:**
  - Saves predictions in a structured CSV file for easy interpretation and further analysis.
  - Outputs detailed logs for monitoring and debugging.

- **Logging and Error Handling:**
  - Integrates a logging system to provide real-time feedback on pipeline execution.
  - Implements a custom exception handling mechanism to manage errors gracefully.

## Usage

### Requirements

- Python 3.x
- Required libraries:
    - pandas
    - joblib
    - scikit-learn
    - xgboost
    - feather-format (or any other necessary library)

### Installation

#### 1. Clone the repository:
        
        git clone https://github.com/your_username/project_name.git
        cd project_name
   

#### 2. Data Preprocessing and Model Training

* To train the model, follow these steps:
* Load the training dataset and preprocess it.
* Train the machine learning model using the specified algorithm.
* Save the trained model and preprocessor to the artifacts folder.

        src/components/data_ingestion.py

#### 3. Prediction

#### To make predictions, the following steps are performed:

**Load the Model and Preprocessor:** Loads the saved model and preprocessor from the artifacts directory.

**Load Data:** Reads the test dataset from a specified .ftr file and processes it to match the model's input requirements.

**Data Transformation:** Transforms the test data using the preprocessor.

**Make Predictions:** Utilizes the trained model to generate predictions based on the transformed data.
**Save Results:** Stores the prediction results in predictions.csv.

Run the prediction script:

        src/pipeline/predict_pipeline.py


