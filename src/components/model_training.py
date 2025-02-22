import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor, XGBClassifier  # ✅ Added XGBClassifier

from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier  # ✅ Added classifiers
)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,  # ✅ Added missing Ridge, Lasso, ElasticNet
    LogisticRegression  # ✅ Added missing Logistic Regression
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier  # ✅ Added DecisionTreeClassifier
from sklearn.svm import SVR, SVC  # ✅ Added Support Vector Models
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier  # ✅ Added KNN Classifier
from sklearn.naive_bayes import GaussianNB  # ✅ Added Naive Bayes
from sklearn.metrics import r2_score

from src.exception import ProjectException
from src.logging import logging
from src.utils import save_object, evaluate_models


# configaration class for model training 

@dataclass
class modeltraining_confi:
    #defining the path where the best trianed model will be saved
    trained_model_file_path = os.path.join('artifacts',"model.pkl")


# model trainer process 
class modelTrainer:
    def __init__(self):
        self.modeltraining_confi = modeltraining_confi()

# model training precocess 
    def initiate_model_trainer(self,train_arry, test_arry):
        try:
            logging.info("spliting train and test imput data")
            X_train, y_train, X_test, y_test = (
            train_arry[:, :-1],
            train_arry[:, -1],
            test_arry[:, :-1],
            test_arry[:, -1],
            )

            # List of models 
                # 📌 Updated List of Models
            models = {
                # 📌 Regression Models
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),  # ✅ Added
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "Support Vector Regressor (SVR)": SVR(),
                "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
                "CatBoost Regressor": CatBoostRegressor(),  # ✅ Added missing CatBoost

                # 📌 Classification Models
                "Logistic Regression": LogisticRegression(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),  # ✅ Added
                "AdaBoost Classifier": AdaBoostClassifier(),
                "XGBoost Classifier": XGBClassifier(),
                "Support Vector Classifier (SVC)": SVC(),
                "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
                "Naive Bayes (GaussianNB)": GaussianNB()
            }
#4.3. Defining Hyperparameters for Each Model
            # Fixed Hyperparameter Grids

            #-----------------Classifier------------------------------
            """
            hyperparameter_grids = {
                "Logistic Regression": {
                "penalty": ["l1", "l2", "elasticnet", None],  
                "C": [0.01, 0.1, 1, 10, 100],  # Regularization strength
                "solver": ["liblinear", "saga"],  # Optimizers
                "max_iter": [100, 200, 500]
            },

            "Decision Tree Classifier": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"],  # ✅ Classification uses "gini" or "entropy"
                },

            "Random Forest Classifier": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"],  # ✅ Classification uses "gini" or "entropy"
                },

            "Gradient Boosting Classifier": {
                "n_estimators": [50, 100, 200, 500],  
                "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                "max_depth": [3, 5, 10],  
                "subsample": [0.8, 1.0],  
                "random_state": [42]
            },

            "XGBoost Classifier": {
                "n_estimators": [50, 100, 200, 500],  
                "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                "max_depth": [3, 5, 10],  
                "subsample": [0.8, 1.0],  
                "colsample_bytree": [0.8, 1.0],  
                "gamma": [0, 0.1, 0.3],  
                "random_state": [42]
            },

            "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1, 10],
                    "estimator": [DecisionTreeClassifier(max_depth=3), None],  # ✅ Use "estimator" instead of "base_estimator"
                },

            "Support Vector Classifier (SVC)": {
                "C": [0.1, 1, 10, 100],  # Regularization
                "kernel": ["linear", "poly", "rbf", "sigmoid"],  
                "gamma": ["scale", "auto"],  # Kernel coefficient
                "degree": [3, 5, 7],  # For polynomial kernel
                "random_state": [42]
            },

            "K-Nearest Neighbors (KNN) Classifier": {
                "n_neighbors": [3, 5, 7, 10],  
                "weights": ["uniform", "distance"],  
                "metric": ["euclidean", "manhattan", "minkowski"]
            },

            "Naive Bayes (GaussianNB)": {
                "var_smoothing": [1e-9, 1e-8, 1e-7]
            },
                #-----------------Regression------------------------------

            "Linear Regression": {
                "fit_intercept": [True, False],  
                "copy_X": [True, False],  # Controls whether X is copied before fitting
                "n_jobs": [-1, None]  # Use multiple CPU cores (-1 for all cores)
            },

            "Ridge Regression": {
                "alpha": [0.01, 0.1, 1, 10, 100],  # Regularization strength
                "solver": ["auto", "svd", "cholesky", "lsqr"]
            },

            "Lasso Regression": {
                "alpha": [0.01, 0.1, 1, 10, 100],  
                "max_iter": [1000, 5000, 10000]
            },

            "ElasticNet Regression": {
                "alpha": [0.01, 0.1, 1, 10],  
                "l1_ratio": [0.1, 0.5, 0.7, 1.0],  
                "max_iter": [1000, 5000]
            },

            "Decision Tree Regressor": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],  # ✅ Fixed
                },

            "Random Forest Regressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],  # ✅ Fixed values
                },

            "Gradient Boosting Regressor": {
                "n_estimators": [50, 100, 200, 500],  
                "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                "max_depth": [3, 5, 10],  
                "subsample": [0.8, 1.0]
            },

            "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1, 10],
                    "loss": ["linear", "square", "exponential"],  # ✅ Valid loss options
                    "estimator": [DecisionTreeRegressor(max_depth=3), None],  # ✅ Use "estimator" instead of "base_estimator"
                },

            "XGBoost Regressor": {
                "n_estimators": [50, 100, 200, 500],  
                "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                "max_depth": [3, 5, 10],  
                "subsample": [0.8, 1.0],  
                "colsample_bytree": [0.8, 1.0],  
                "gamma": [0, 0.1, 0.3]
            },

            "Support Vector Regressor (SVR)": {
                "C": [0.1, 1, 10, 100],  
                "kernel": ["linear", "poly", "rbf", "sigmoid"],  
                "gamma": ["scale", "auto"],  
                "degree": [3, 5, 7]
            },

            "K-Nearest Neighbors (KNN) Regressor": {
                "n_neighbors": [3, 5, 7, 10],  
                "weights": ["uniform", "distance"],  
                "metric": ["euclidean", "manhattan", "minkowski"]
            }

            }

            """
            # ----------------- Classifier ------------------------------
            hyperparameter_grids = {
                "Logistic Regression": {
                    "penalty": ["l2"],  # Simplify to only L2 penalty
                    "C": [0.1, 1, 10],  # Reduce the range of regularization strength
                    "solver": ["liblinear"],  # Only use "liblinear" for smaller datasets
                    "max_iter": [100]
                },
                
                "Decision Tree Classifier": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "criterion": ["gini"]
                },

                "Random Forest Classifier": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "criterion": ["gini"]
                },

                "Gradient Boosting Classifier": {
                    "n_estimators": [50, 100],  
                    "learning_rate": [0.1, 0.2],  
                    "max_depth": [3, 5],  
                    "subsample": [0.8]
                },

                "XGBoost Classifier": {
                    "n_estimators": [50, 100],  
                    "learning_rate": [0.1, 0.2],  
                    "max_depth": [3, 5],  
                    "subsample": [0.8],  
                    "colsample_bytree": [0.8]
                },

                "AdaBoost Classifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1, 1]
                },

                "Support Vector Classifier (SVC)": {
                    "C": [1, 10],
                    "kernel": ["linear", "rbf"],  
                    "gamma": ["scale"]
                },

                "K-Nearest Neighbors (KNN) Classifier": {
                    "n_neighbors": [3, 5],
                    "weights": ["uniform"],
                    "metric": ["euclidean"]
                },

                "Naive Bayes (GaussianNB)": {
                    "var_smoothing": [1e-9, 1e-8]
                },

                "Lasso": {"alpha": [0.1, 1.0, 10.0]
                
                },

                # ----------------- Regression ------------------------------
                
                "Linear Regression": {
                    "fit_intercept": [True],  
                    "copy_X": [True],  
                    "n_jobs": [-1]
                },

                "Ridge Regression": {
                    "alpha": [0.1, 1, 10],  
                    "solver": ["auto"]
                },

                "Lasso Regression": {
                    "alpha": [0.1, 1],  
                    "max_iter": [1000]
                },

                "ElasticNet Regression": {
                    "alpha": [0.1, 1],  
                    "l1_ratio": [0.5],  
                    "max_iter": [1000]
                },

                "Decision Tree Regressor": {
                    "max_depth": [None, 10],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                    "criterion": ["squared_error"]
                },

                "Random Forest Regressor": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                    "criterion": ["squared_error"]
                },

                "Gradient Boosting Regressor": {
                    "n_estimators": [50, 100],  
                    "learning_rate": [0.1],  
                    "max_depth": [3, 5],  
                    "subsample": [0.8]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1],
                    "loss": ["linear"]
                },

                "XGBoost Regressor": {
                    "n_estimators": [50, 100],  
                    "learning_rate": [0.1],  
                    "max_depth": [3],  
                    "subsample": [0.8]
                },

                "Support Vector Regressor (SVR)": {
                    "C": [1, 10],  
                    "kernel": ["linear", "rbf"],  
                    "gamma": ["scale"]
                },

                "K-Nearest Neighbors (KNN) Regressor": {
                    "n_neighbors": [3, 5],
                    "weights": ["uniform"],
                    "metric": ["euclidean"]
                },
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "ElasticNet": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}
            }

# Fixed Hyperparameter Grids for RandomizedSearchCV

            """
            hyperparameter_grids = {
                "Logistic Regression": {
                "penalty": ["l1", "l2", "elasticnet", None],  
                "C": [0.01, 0.1, 1, 10, 100],  # Regularization strength
                "solver": ["liblinear", "saga"],  # Optimizers
                "max_iter": [100, 200, 500]
            },

            "Decision Tree Classifier": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"],  # ✅ Classification uses "gini" or "entropy"
                },

            "Random Forest Classifier": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"],  # ✅ Classification uses "gini" or "entropy"
                },

            "Gradient Boosting Classifier": {
                "n_estimators": [50, 100, 200, 500],  
                "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                "max_depth": [3, 5, 10],  
                "subsample": [0.8, 1.0],  
                "random_state": [42]
            },

            "XGBoost Classifier": {
                "n_estimators": [50, 100, 200, 500],  
                "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                "max_depth": [3, 5, 10],  
                "subsample": [0.8, 1.0],  
                "colsample_bytree": [0.8, 1.0],  
                "gamma": [0, 0.1, 0.3],  
                "random_state": [42]
            },

            "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.5, 1.0],
                    "estimator": [DecisionTreeClassifier(max_depth=3), None],  # ✅ Use "estimator" instead of "base_estimator"
                },

            "Support Vector Classifier (SVC)": {
                "C": [0.1, 1, 10, 100],  # Regularization
                "kernel": ["linear", "poly", "rbf", "sigmoid"],  
                "gamma": ["scale", "auto"],  # Kernel coefficient
                "degree": [3, 5, 7],  # For polynomial kernel
                "random_state": [42]
            },

            "K-Nearest Neighbors (KNN) Classifier": {
                "n_neighbors": [3, 5, 7, 10],  
                "weights": ["uniform", "distance"],  
                "metric": ["euclidean", "manhattan", "minkowski"]
            },

            "Naive Bayes (GaussianNB)": {
                "var_smoothing": [1e-9, 1e-8, 1e-7]
            },
                #-----------------Regression------------------------------

            "Linear Regression": {
                "fit_intercept": [True, False],  
                "copy_X": [True, False],  
                "n_jobs": [None, -1],  # Use all CPU cores if possible
                "positive": [True, False]  # Force positive coefficients
            },

            "Ridge Regression": {
                "alpha": [0.01, 0.1, 1, 10, 100],  # Regularization strength
                "solver": ["auto", "svd", "cholesky", "lsqr"]
            },

            "Lasso Regression": {
                "alpha": [0.01, 0.1, 1, 10, 100],  
                "max_iter": [1000, 5000, 10000]
            },

            "ElasticNet Regression": {
                "alpha": [0.01, 0.1, 1, 10],  
                "l1_ratio": [0.1, 0.5, 0.7, 1.0],  
                "max_iter": [1000, 5000]
            },

            "Decision Tree Regressor": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],  # ✅ Fixed
                },

            "Random Forest Regressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],  # ✅ Fixed values
                },

            "Gradient Boosting Regressor": {
                "n_estimators": [50, 100, 200, 500],  
                "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                "max_depth": [3, 5, 10],  
                "subsample": [0.8, 1.0]
            },

            "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.5, 1.0],
                    "loss": ["linear", "square", "exponential"],  # ✅ Use only valid loss functions
                    "estimator": [DecisionTreeRegressor(max_depth=3), None],  # ✅ Use "estimator" instead of "base_estimator"
                },

            "XGBoost Regressor": {
                "n_estimators": [50, 100, 200, 500],  
                "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                "max_depth": [3, 5, 10],  
                "subsample": [0.8, 1.0],  
                "colsample_bytree": [0.8, 1.0],  
                "gamma": [0, 0.1, 0.3]
            },

            "Support Vector Regressor (SVR)": {
                "C": [0.1, 1, 10, 100],  
                "kernel": ["linear", "poly", "rbf", "sigmoid"],  
                "gamma": ["scale", "auto"],  
                "degree": [3, 5, 7]
            },

            "K-Nearest Neighbors (KNN) Regressor": {
                "n_neighbors": [3, 5, 7, 10],  
                "weights": ["uniform", "distance"],  
                "metric": ["euclidean", "manhattan", "minkowski"]
            }
            }

            """
            # ----------------- Classifier ------------------------------
            hyperparameter_grids = {
                "Logistic Regression": {
                    "penalty": ["l2"],  # Only L2 penalty
                    "C": [0.1, 1, 10],  # Regularization strength
                    "solver": ["liblinear"],  # Use a single solver for efficiency
                    "max_iter": [100]
                },
                
                "Decision Tree Classifier": {
                    "max_depth": [None, 10],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                    "criterion": ["gini"]
                },

                "Random Forest Classifier": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                    "criterion": ["gini"]
                },

                "Gradient Boosting Classifier": {
                    "n_estimators": [50, 100],  
                    "learning_rate": [0.1],  
                    "max_depth": [3, 5],  
                    "subsample": [0.8]
                },

                "XGBoost Classifier": {
                    "n_estimators": [50, 100],  
                    "learning_rate": [0.1],  
                    "max_depth": [3, 5],  
                    "subsample": [0.8],  
                    "colsample_bytree": [0.8]
                },

                "AdaBoost Classifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1]
                },

                "Support Vector Classifier (SVC)": {
                    "C": [1, 10],
                    "kernel": ["linear", "rbf"],  
                    "gamma": ["scale"]
                },

                "K-Nearest Neighbors (KNN) Classifier": {
                    "n_neighbors": [3, 5],
                    "weights": ["uniform"],
                    "metric": ["euclidean"]
                },

                "Naive Bayes (GaussianNB)": {
                    "var_smoothing": [1e-9]
                },

                # ----------------- Regression ------------------------------
                
                "Linear Regression": {
                    "fit_intercept": [True],  
                    "copy_X": [True],  
                    "n_jobs": [-1]
                },

                "Ridge Regression": {
                    "alpha": [0.1, 1, 10],  
                    "solver": ["auto"]
                },

                "Lasso Regression": {
                    "alpha": [0.1, 1],  
                    "max_iter": [1000]
                },

                "ElasticNet Regression": {
                    "alpha": [0.1, 1],  
                    "l1_ratio": [0.5],  
                    "max_iter": [1000]
                },

                "Decision Tree Regressor": {
                    "max_depth": [None, 10],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                    "criterion": ["squared_error"]
                },

                "Random Forest Regressor": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                    "criterion": ["squared_error"]
                },

                "Gradient Boosting Regressor": {
                    "n_estimators": [50, 100],  
                    "learning_rate": [0.1],  
                    "max_depth": [3],  
                    "subsample": [0.8]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1],
                    "loss": ["linear"]
                },

                "XGBoost Regressor": {
                    "n_estimators": [50, 100],  
                    "learning_rate": [0.1],  
                    "max_depth": [3],  
                    "subsample": [0.8]
                },

                "Support Vector Regressor (SVR)": {
                    "C": [1, 10],  
                    "kernel": ["linear", "rbf"],  
                    "gamma": ["scale"]
                },

                "K-Nearest Neighbors (KNN) Regressor": {
                    "n_neighbors": [3, 5],
                    "weights": ["uniform"],
                    "metric": ["euclidean"]
                },
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.1, 1.0, 10.0]},
                "ElasticNet": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]},
                
            }

            # evaluate all the modesl 
            model_report: dict = evaluate_models(
                X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                models = models, param=hyperparameter_grids
            )

            #Selecting the Best Model
            # Filter out models that failed to fit
            valid_model_scores = [score for score in model_report.values() if isinstance(score, (int, float))]
            # If no valid models are found, handle it gracefully
            if not valid_model_scores:
                raise ValueError("No models were successfully trained.")
            # Find the best model score
            best_model_score = max(valid_model_scores)
            # Get the corresponding model name
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]

            best_model = models[best_model_name]  # ✅ Get the trained model instance

            #Handling Cases Where No Good Model is Found
            if best_model_score < 0.6:
                raise Exception("no best model found")
            
            # Save the trained model, not just its name
            save_object(
                file_path=self.modeltraining_confi.trained_model_file_path,
                obj=best_model  # ✅ Save the model, not the string name
            )

            # Make predictions using the best trained model
            predicted = best_model.predict(X_test)  # ✅ Use the trained model to predict
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise ProjectException(e, sys)
