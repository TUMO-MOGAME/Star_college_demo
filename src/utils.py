import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from src.exception import ProjectException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise ProjectException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})  # Use .get() to avoid KeyError

            # 🚀 Skip GridSearchCV for CatBoost (or other unsupported models)
            if model_name == "CatBoost Regressor":
                print(f"Skipping GridSearchCV for {model_name} (not fully compatible)")
                model.fit(X_train, y_train)  # Train model directly
            else:
                try:
                    gs = GridSearchCV(model, para, cv=3)
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)  # Apply best params
                except Exception as e:
                    print(f"⚠️ Skipping GridSearchCV for {model_name} due to error: {e}")
                    print(f"Training {model_name} without hyperparameter tuning...")
                
            # Ensure every model is fit before prediction
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"❌ Model {model_name} could not be fit: {e}")
                report[model_name] = "Model not fit"
                continue  # Skip prediction for this model

            # 🚀 Model is now trained, proceed with predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise ProjectException(e, sys)


    except Exception as e:
        raise ProjectException(e, sys)


    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise ProjectException(e, sys)