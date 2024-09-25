import pandas as pd
import os
import sys
import joblib
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging



@dataclass
class PredictionPipelineConfig:
    preprocessor_path: str=os.path.join('artifacts', 'preprocessor.pkl')
    model_path: str=os.path.join('artifacts', 'model.pkl')
    predict_data_path: str=os.path.join('notebook/data', 'test.ftr')
    predictions_output_path: str = os.path.join('artifacts', 'predictions.csv')

class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()
   # model ve preprocessori yukle
    def load_preprocessor_and_model(self):
        try:
            self.preprocessor = joblib.load(self.config.preprocessor_path)
            self.model = joblib.load(self.config.model_path)
            logging.info(f'Preprocessor and model loaded')
        except Exception as e:
            raise CustomException(e,sys)



    def load_data(self):
        try:
            self.data = pd.read_feather(self.config.predict_data_path)
            df = self.data.groupby('customer_ID').tail(1).set_index('customer_ID')
            df = df.drop('Spend_2', axis=1)
            df = df.head(1)
            logging.info("'customer_ID' column set as index")

            self.data = df
            return self.data
        except Exception as e:
            raise CustomException(e,sys)


    def predict(self, processed_data):
        try:
            logging.info('Making predictions using the model')
            predictions = self.model.predict(processed_data)  # Model ile tahmin yap
            logging.info(f'Predictions completed: {predictions}')
            return predictions
        except Exception as e:
            raise CustomException(e,sys)

    def process_new_data(self):
        try:
            self.load_data()
            processed_data = self.preprocessor.transform(self.data)
            predictions = self.predict(processed_data)
            self.save_predictions(predictions)
            return predictions
        except Exception as e:
            raise CustomException(e,sys)


    def save_predictions(self, predictions):
        try:
            predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
            predictions_df.to_csv(self.config.predictions_output_path, index=False)
            logging.info(f'Predictions saved to {self.config.predictions_output_path}')
        except Exception as e:
            raise CustomException(e, sys)



if __name__ == '__main__':

    pipeline = PredictionPipeline()
    pipeline.load_preprocessor_and_model()
    result = pipeline.process_new_data()

