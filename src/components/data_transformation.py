import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, df: pd.DataFrame) ->pd.DataFrame:
        try:
            categorical_columns = df.select_dtypes(include=['category']).columns.tolist()

            numerical_columns = df.select_dtypes(include=['float16']).columns.tolist()

            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value="Other")),
                ('onehotencoder', OneHotEncoder())
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', KNNImputer())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                    ("num_pipelines", num_pipeline, numerical_columns)

                ]
            )

            logging.info(f"Pipeline processed")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_feather(train_path)
            test_df=pd.read_feather(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object(train_df)

            target_column_name="target"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info(f"Input feature train array shape: {input_feature_train_arr.shape}")

            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info(f"Input feature test array shape: {input_feature_test_arr.shape}")

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]

            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)










