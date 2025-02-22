import os 
import sys
import pandas as pd
import numpy as np

from src.exception import ProjectException
from src.logging import logging

class datavalidation:
    def __init__(self,train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self,file_path):
        """load data from a given file."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ProjectException(f"Erro loading data from{file_path}: {str(e)},sys")
        
    def checking_missing_values(self,df):
        """Checking for missing values in the datase."""
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values/len(df)) * 100
        return (missing_values[missing_values > 0],
               missing_percentage[missing_percentage > 0]
               )
    def check_data_types(self,df):
        """Cheking for inconsistenf data types."""
        return df.dtypes
    
    def check_duplicates(self,df):
        """checking for duplicates rows"""
        return df.duplicated().sum()
    
    def check_outliers(self, df, threshold=3):
        """Detect outliers using Z-score."""
        numerical_cols = df.select_dtypes(include=['number']).columns
        z_scores = np.abs((df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std())
        return (z_scores > threshold).sum()
    
    def check_unique_values(self, df):
        """Check unique values for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        return {col: df[col].nunique() for col in categorical_cols}

    def validate_data(self):
        """Perform all validation checks on train and test datasets."""
        try:
            logging.info("Starting data validation process...")

            train_df = self.load_data(self.train_path)
            test_df = self.load_data(self.test_path)

            for dataset, name in [(train_df, "Train"), (test_df, "Test")]:
                logging.info(f"Validating {name} dataset...")

                # Missing values
                missing_vals, missing_perc = self.checking_missing_values(dataset)
                logging.info(f"Missing values in {name} dataset:\n{missing_vals}\nPercentage:\n{missing_perc}")

                if not missing_vals.empty:
                    logging.error(f"Validation failed: Missing values detected in {name} dataset.")
                    return False  # 🚨 Return False if validation fails

                # Duplicates
                duplicates = self.check_duplicates(dataset)
                logging.info(f"Duplicate rows in {name} dataset: {duplicates}")

                # Data types
                data_types = self.check_data_types(dataset)
                logging.info(f"Data types in {name} dataset:\n{data_types}")

                # Outliers
                outliers = self.check_outliers(dataset)
                logging.info(f"Outliers detected in {name} dataset:\n{outliers}")

                # Unique values
                unique_values = self.check_unique_values(dataset)
                logging.info(f"Unique values in categorical columns ({name} dataset):\n{unique_values}")

            logging.info("Data validation process completed successfully.")
            return True  # ✅ Return True when validation is successful


            logging.info("Data validation process completed.")
        except Exception as e:
            raise ProjectException(e, sys)
