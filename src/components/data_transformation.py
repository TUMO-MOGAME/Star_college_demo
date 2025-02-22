import os 
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import ProjectException
from src.logging import logging

from src.utils import save_object

#data transformation config class

@dataclass
class data_transformation_config:
    preprocesssor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

#data transformation class 
class dataTransformation:
    def __init__(self):
        self.data_transformation_config = data_transformation_config()
    
    #this funstion is responsible for data transformation
    # it defines how both numerical and categorical data should be handled

    def initialize_dataTransformation(self):
        try:
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = [
                "gender","race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
                ]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),  # Use "mean" instead of "most_frequent" for numerical columns
                    ("scaler", StandardScaler(with_mean=False))  # Standardize numerical features
                        ]
                )


            categorical_pipeline = Pipeline(
                steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing categorical values
                        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),  # ✅ Ignore unknown categories
                        ("scaler", StandardScaler(with_mean=False))  # Standardize encoded values
                        ]
                    )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #columnTransformer:
            

            preprocessor = ColumnTransformer([
                ("numerical_pipeline", numerical_pipeline, numerical_columns),
                ("categorical_pipeline", categorical_pipeline, categorical_columns)
            ])


            #returning the preprocessor object:
            return preprocessor
        
        except Exception as e:
            raise ProjectException(e,sys)
        
    # start data transformation
    def start_data_transformation(self,train_path,test_path):
        try:
            #reading the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("reading train data and test data complete")

            print("Columns BEFORE dropping target column:", train_df.columns.tolist())
            #defining the target and features columns
            target_column = "math_score"
            #spliting feature and target
            x_train = train_df.drop(columns=[target_column], axis = 1)
            y_train = train_df[target_column]
            print("Columns AFTER dropping target column:", x_train.columns.tolist())

            print("Columns BEFORE dropping target column:", train_df.columns.tolist())
            x_test = test_df.drop(columns=[target_column], axis =1)
            y_test = test_df[target_column]
            print("Columns AFTER dropping target column:", x_train.columns.tolist())

            #obtaining the preprocessing object
            logging.info("obtaining preprocessing object")
            preprocessor = self.initialize_dataTransformation()

            # Applying preprocessing to training and testing data
            logging.info("Applying preprocessing object on training and testing data")
            x_train_arr = preprocessor.fit_transform(x_train)
            x_test_arr = preprocessor.transform(x_test)

            # Combining Features and Target:
            train_arr = np.c_[x_train_arr, np.array(y_train)]
            test_arr = np.c_[x_test_arr, np.array(y_test)]

            # Saving the Preprocessor Object:
            save_object(
                file_path=self.data_transformation_config.preprocesssor_obj_file_path,
                obj=preprocessor
            )

            # Returning the transformed data and preprocessor object path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesssor_obj_file_path,
            )
        
        except Exception as e:
            raise ProjectException(e,sys)

