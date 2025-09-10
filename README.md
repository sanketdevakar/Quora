## Quora Question Duplicates Detection

**Problem Statement**
- Identify which questions asked on Quora are duplicates of questions that have already been asked.
- This is useful to instantly provide answers to questions that have already been answered, improving user experience and reducing redundancy.
- The task is framed as a binary classification problem: given a pair of questions, predict whether they are duplicates (1) or not (0).

**Dataset**
- The dataset contains pairs of questions and a target column is_duplicate (1 if duplicate, 0 otherwise).
- Features are numerical embeddings extracted from text preprocessing or external embeddings.
- Original class distribution: imbalanced, with more non-duplicate pairs (class 0) than duplicate pairs (class 1)

## Workflow: Quora Question Duplicates Detection

1. **Data Ingestion**  
   - Load train and test CSV files.  
   - Inspect data and check for missing or non-numeric features.  

2. **Data Preprocessing**  
   - Drop non-numeric columns.  
   - Separate features (`X`) and target (`y`).  
      
3. **Feature Engineering**  
   - Convert text questions into numerical representations. 
   - Techniques include:
      - TF-IDF: Capture importance of words in question pairs.
      - Word2Vec embeddings: Represent each question as dense vector capturing semantic meaning.
      - Combine TF-IDF and Word2Vec features for better representation of question similarity. 



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
