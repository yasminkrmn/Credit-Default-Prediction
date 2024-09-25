import os
import sys
from dataclasses import dataclass
import yaml

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model



@dataclass
class ModelTrainerConfig():
    trained_model_obj_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f'Split training and test input data')
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1],
                                                test_array[:,:-1], test_array[:,-1])

            with open('src/components/params.yaml', 'r') as file:
                params = yaml.safe_load(file)

            models = {
                'RandomForestClassifier': RandomForestClassifier(class_weight='balanced'),
                'LGBMClassifier': lgb.LGBMClassifier()
            }




            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train,
                                               X_test=X_test, y_test=y_test,
                                               models=models, params=params)

            logging.info(f'model_report:\n{model_report}')

            best_model_name = None
            best_model_score = -1

            for model_name, metrics in model_report.items():
                test_auc_score = metrics['Test AUC Score']
                if test_auc_score > best_model_score:
                    best_model_score = test_auc_score
                    best_model_name = model_name

            logging.info(f'Best model name: {best_model_name}')
            logging.info(f'Best model score: {best_model_score}')

            best_model = models[best_model_name]
            logging.info(f'Best model: {best_model}')

            best_model.fit(X_train, y_train)
            logging.info('Fitting the best model completed.')

            if best_model_score < 0.5:
                raise CustomException('No best model found')
            logging.info(f'Best model found for train and test')

            save_object(
                file_path=self.model_trainer_config.trained_model_obj_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)







