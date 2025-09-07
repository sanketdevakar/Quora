import sys
import os
import pandas as pd
from src.logger import  logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv('notebook/data/train.csv')  # path to raw CSV

            # Remove duplicates
            df = df.drop_duplicates().reset_index(drop=True)

            # Filling the null values with ' '
            df = df.fillna('')

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False)

            # Split train-test

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed")

            return (
                    self.config.train_data_path, 
                    self.config.test_data_path
            ) 
        
        except Exception as e:
            raise CustomException(e,sys)

