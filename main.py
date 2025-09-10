from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import time
import sys
import threading

def heartbeat():
    """Background heartbeat to show script is alive."""
    while True:
        logging.info("Script is still running...")
        time.sleep(60)

if __name__ == "__main__":
    logging.info("✅ Script execution started")

    # Start heartbeat thread
    threading.Thread(target=heartbeat, daemon=True).start()

    try:
        # # Data ingestion
        # logging.info("🚀 Data Ingestion started...")
        # ingestion = DataIngestion()
        # train_path, test_path = ingestion.initiate_data_ingestion()
        # logging.info("✅ Data Ingestion completed")

        # # Data transformation
        # logging.info("🔄 Data Transformation started...")
        # transformer = DataTransformation()
        # transformed_train, transformed_test, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)
        # logging.info("✅ Data Transformation completed")
        print("Training the model")
        trainer = ModelTrainer()
        model_path = trainer.initiate_model_training(
                        "artifacts/train_transformed.csv",
                        "artifacts/test_transformed.csv"
                        
                    )

    except Exception as e:
        logging.error(f"❌ Script failed with error: {e}")
        raise CustomException(e,sys)

    logging.info("🎯 Script execution finished successfully")