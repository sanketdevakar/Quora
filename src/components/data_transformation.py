# src/components/data_transformation.py
import sys
import os
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
from src.utils import preprocess, add_handcrafted_features, add_spacy_embeddings, expand_vector_columns

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = "artifacts/preprocessor.pkl"
    transformed_train_path: str = "artifacts/train_transformed.csv"
    transformed_test_path: str = "artifacts/test_transformed.csv"

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):

        logging.info("Starting data transformation")
        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print("loaded the data")
            #  Clean text
            train_df['question1'] = train_df['question1'].apply(preprocess)
            train_df['question2'] = train_df['question2'].apply(preprocess)
            test_df['question1'] = test_df['question1'].apply(preprocess)
            test_df['question2'] = test_df['question2'].apply(preprocess)
            print("adding handcrafted feature")
            #   Handcrafted features
            train_df = add_handcrafted_features(train_df)
            test_df = add_handcrafted_features(test_df)

            logging.info("Embedding with spacy and tf-idf has started")
            
            #  SpaCy embeddings
            train_df = add_spacy_embeddings(train_df)
            test_df = add_spacy_embeddings(test_df)
            print(" Ended spacy embeddings")
            #  Expand embeddings into separate columns
            train_df = expand_vector_columns(train_df, "q1_feats_m")
            train_df = expand_vector_columns(train_df, "q2_feats_m")
            test_df = expand_vector_columns(test_df, "q1_feats_m")
            test_df = expand_vector_columns(test_df, "q2_feats_m")
            print("Preprocessing the numerical feature")
            #  Identify numeric features (exclude IDs and target)
            features_to_exclude = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
            numeric_features = [c for c in train_df.columns if c not in features_to_exclude]

            #  Scale numeric features
            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numeric_features)
            ])
            preprocessor.fit(train_df[numeric_features])
            print("Applying scaling to the data")
            # Apply scaling
            train_df[numeric_features] = preprocessor.transform(train_df[numeric_features])
            test_df[numeric_features] = preprocessor.transform(test_df[numeric_features])
            print("Saving preprocessor.pkl")
            #  Save preprocessor and transformed data
            with open(self.config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            train_df.to_csv(self.config.transformed_train_path, index=False)
            print("Added train data")
            test_df.to_csv(self.config.transformed_test_path, index=False)
            print("Added test data")
            logging.info("Data transformation completed")

            return(self.config.transformed_train_path,
                    self.config.transformed_test_path, 
                    self.config.preprocessor_obj_file_path) 
        

        except Exception as e:
            raise CustomException(e,sys)