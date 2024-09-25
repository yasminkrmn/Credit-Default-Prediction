import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


from src.exception import CustomException

def save_object(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except CustomException as e:
        raise CustomException(e, sys)


def gini_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


def evaluate_model(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(estimator=model, param_grid=param, cv=10, n_jobs=-1)
            gs.fit(X_train, y_train)
            # model.fit(X_train, y_train)
            # model.set_params(**params)
            best_model = gs.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)


            train_model_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
            test_model_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

            y_train_pred_df = pd.DataFrame({'target': y_train, 'prediction': y_train_pred})
            y_test_pred_df = pd.DataFrame({'target': y_test, 'prediction':y_test_pred})

            gini_score_train = gini_metric(y_train_pred_df[['target']], y_train_pred_df[['prediction']])
            gini_score_test = gini_metric(y_test_pred_df[['target']], y_test_pred_df[['prediction']])

            report[list(models.keys())[i]] = {
                "Train AUC Score": train_model_auc,
                "Test AUC Score": test_model_auc,
                "Train Gini Metric": gini_score_train,
                "Test Gini Metric": gini_score_test
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
