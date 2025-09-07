from src.logger import  logging
from src.components.data_ingestion import DataIngestion


logging.info("Data Ingestion has started")

if __name__ == "__main__":
    #  Data ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()