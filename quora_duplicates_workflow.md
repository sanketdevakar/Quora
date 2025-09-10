## Workflow: Quora Question Duplicates Detection

1. **Data Ingestion**  
   - Load train and test CSV files.  
   - Inspect data and check for missing or non-numeric features.  

2. **Data Preprocessing**  
   - Drop non-numeric columns.  
   - Separate features (`X`) and target (`y`).  
   - Optionally limit training size with `max_train_size`.  

3. **Handle Class Imbalance**  
   - Oversample minority class (duplicates) using `resample`.  
   - Compute dynamic `scale_pos_weight` for XGBoost.  

4. **Feature Scaling**  
   - Scale numeric features for linear models (Logistic Regression, LinearSVM).  
   - Tree-based models (RandomForest, XGBoost) use raw features.  

5. **Model Training**  
   - Candidate models:  
     - RandomForest (balanced)  
     - Logistic Regression (balanced)  
     - LinearSVM (balanced)  
     - XGBoost (hyperparameter tuned, balanced)  
   - Hyperparameter tuning for XGBoost via `RandomizedSearchCV`.  

6. **Model Evaluation**  
   - Predict on test set.  
   - Evaluate using precision, recall, F1-score, accuracy.  
   - Compare models and select **best model based on micro F1-score**.  

7. **Save Best Model**  
   - Save the trained model and scaler as a pickle file (`best_model.pkl`).  

8. **Inference**  
   - Load saved model and scaler.  
   - Preprocess new question pairs the same way.  
   - Predict whether a pair is duplicate (`1`) or not (`0`).