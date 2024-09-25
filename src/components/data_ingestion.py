import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.ftr')
    test_data_path: str=os.path.join('artifacts', 'test.ftr')
    raw_data_path: str=os.path.join('artifacts', 'raw.ftr')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info(f'Initiating data ingestion method or component')
        try:
            df=pd.read_feather('notebook/data/train.ftr')

            logging.info(f'Read the dataset as dataframe')

            df = df.groupby('customer_ID').tail(1).set_index('customer_ID')
            df = df.drop('Spend_2', axis=1)
            logging.info(f"'customer_ID' column set as index")


            if df['target'].isnull().any():
                logging.warning('NaN values found in target column. Dropping these rows.')
                df = df.dropna(subset=['target'])

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_feather(self.ingestion_config.raw_data_path)

            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

            train_set.to_feather(self.ingestion_config.train_data_path)

            test_set.to_feather(self.ingestion_config.test_data_path)

            logging.info(f'Train and test data ingestion completed')

            return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer=ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr, test_arr)

