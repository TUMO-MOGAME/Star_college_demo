import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logging import logging
from src.exception import ProjectException

##from src.components.data_transformation import data_tranformation
##from src.components.data_validation import data_validation
##from src.components.model_training import model_trining 

from sklearn.model_selection import train_test_split

@dataclass
class dataIngestion_config:
    raw_data_path: str=os.path.join('artifacts',"raw_data.csv")
    train_data_path: str=os.path.join('artifacts',"train_data.csv")
    test_data_path: str=os.path.join('artifacts',"test_data.csv")

class dataIngestion:
    def __init__(self):
        self.dataIngestion_config = dataIngestion_config()

    def initialize_dataIngestion(self):
        logging.info("starting with data ingestion")
        try:
            df = pd.read_csv("notebook\\raw_data.csv")

            logging.info("reading the dataset to dataframe")
            df.to_csv(self.dataIngestion_config.raw_data_path, index=False,header=True)
            os.makedirs(os.path.dirname(self.dataIngestion_config.train_data_path),exist_ok=True)

            logging.info("training test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.dataIngestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.dataIngestion_config.test_data_path,index=False,header=True)

            logging.info("end of data ingestion")

            return(
                self.dataIngestion_config.test_data_path,
                self.dataIngestion_config.train_data_path
            )
        except Exception as e:
            raise ProjectException(e,sys)

