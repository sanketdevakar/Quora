import os
import sys
import pandas as pd
from src.logger import logging
import pickle
import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "best_model.pkl")
    max_train_size: int = 100000   # 🔹 configurable sample size


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            
            
            logging.info("Loading transformed train and test data")
            print("Loading Transformed and train data")
            train_df = pd.read_csv(train_array)
            test_df = pd.read_csv(test_array)
            print("seperating numeric features")
            # Separate numeric features and target
            y_train = train_df["is_duplicate"].astype(int)
            y_test = test_df["is_duplicate"].astype(int)
            X_train = train_df.drop("is_duplicate", axis=1).select_dtypes(include=[np.number])
            X_test = test_df.drop("is_duplicate", axis=1).select_dtypes(include=[np.number])

            logging.info(f"Original Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            print("Oversampling minority class in training")
            # --- Oversample minority class in training ---
            train_df_balanced = pd.concat([X_train, y_train], axis=1)
            df_majority = train_df_balanced[train_df_balanced.is_duplicate == 0]
            df_minority = train_df_balanced[train_df_balanced.is_duplicate == 1]

            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=len(df_majority),
                random_state=42
            )

            train_df_balanced = pd.concat([df_majority, df_minority_upsampled])
            train_df_balanced = train_df_balanced.sample(frac=1, random_state=42)  # shuffle
            print("Preparing upsampled training data")
            X_train = train_df_balanced.drop("is_duplicate", axis=1).values
            y_train = train_df_balanced["is_duplicate"].values
            X_test = X_test.values
            y_test = y_test.values

            logging.info(f"Oversampled Train shape: {X_train.shape}")

            # Scale features for LR and SVM
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print("Scaled the data")
            # --- Compute XGBoost scale_pos_weight dynamically ---
            num_class_0 = sum(y_train == 0)
            num_class_1 = sum(y_train == 1)
            scale_pos_weight = num_class_0 / num_class_1
            logging.info(f"XGBoost scale_pos_weight set to: {scale_pos_weight:.2f}")

            # --- Balanced Models ---
            models = {
                "RandomForest_Balanced": RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1
                ),
                "XGBoost_Balanced": XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1
                ),
                "LogisticRegression_Balanced": LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced"
                ),
                "LinearSVM_Balanced": LinearSVC(
                    max_iter=5000,
                    random_state=42,
                    class_weight="balanced"
                ),
            }

            model_results = {}
            best_model_name = None
            best_score = -1
            best_model = None

            for name, model in models.items():
                logging.info(f"Training {name}")
                model.fit(X_train, y_train)

                # Predictions
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                else:
                    y_proba = None
                y_pred = model.predict(X_test)

                # Evaluate
                metrics = evaluate_model(y_test, y_pred, y_proba, plot_cm=False, plot_roc=False)
                model_results[name] = metrics
                print(f"Training is done on model {name}")
                # Choose best model by micro F1
                if metrics["f1_micro"] > best_score:
                    best_score = metrics["f1_micro"]
                    best_model_name = name
                    best_model = model

            logging.info(f"Best Model: {best_model_name} with Micro F1 = {best_score:.4f}")

            # Save best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(best_model, f)

            logging.info("Model training completed and best model saved.")
            return {"best_model": best_model_name, "results": model_results}

        except Exception as e:
            raise CustomException(e, sys)
