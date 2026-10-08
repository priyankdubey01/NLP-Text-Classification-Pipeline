# NLP-Text-Classification-Pipeline
End-to-end NLP text classification using TF-IDF, Naive Bayes, Logistic Regression, and Linear SVM.
A complete Natural Language Processing (NLP) text classification pipeline built with Python and scikit-learn. The project uses the 20 Newsgroups benchmark dataset containing 18,846 documents across 20 categories and demonstrates the complete workflow from text preprocessing to model deployment/inference.
Project Overview
This project implements a classical machine-learning NLP pipeline for multi-class text classification.
Pipeline
1. Load the 20 Newsgroups dataset
2. Explore the dataset
3. Clean and preprocess text using NLTK
4. Split data into training and testing sets
5. Extract TF-IDF features using unigrams and bigrams
6. Train multiple classification models
7. Compare model performance
8. Evaluate the best model using classification metrics and a confusion matrix
9. Save the trained classifier and TF-IDF vectorizer
10. Perform inference on new/unseen text
Dataset
The project uses the 20 Newsgroups dataset through sklearn.datasets.fetch_20newsgroups.
- Documents: 18,846
- Categories: 20
- Training samples: 14,632
- Test samples: 3,658
- Split: 80/20
- Stratification: Enabled
- Random state: 42
The notebook also supports replacing the benchmark dataset with a custom CSV containing a text column and a label column.
Text Preprocessing
The preprocessing pipeline includes:
- Lowercasing
- URL removal
- Email-address removal
- Punctuation removal
- Number removal
- Whitespace normalization
- Tokenization
- English stop-word removal
- WordNet lemmatization
- Removal of very short tokens
Feature Engineering
Text is converted into numerical features using TfidfVectorizer.
Configuration used in the notebook:
max_features = 20,000
ngram_range   = (1, 2)
min_df        = 3
max_df        = 0.9
Resulting feature matrices:
- Training: 14,632 × 20,000
- Testing: 3,658 × 20,000
Machine Learning Models
Three baseline classifiers are trained and compared:
Model	Accuracy
Multinomial Naive Bayes	74.47%
Logistic Regression	75.94%
Linear SVM	76.22%


Best Model
Linear Support Vector Machine (LinearSVC) achieved the highest test accuracy of 76.22% among the evaluated models.
Evaluation
The notebook evaluates the best model using:
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
For the Linear SVM, the reported overall results are approximately:
- Accuracy: 0.76
- Macro-average F1: 0.75
- Weighted-average F1: 0.76
Performance varies by newsgroup category. For example, rec.sport.hockey achieved an F1-score of 0.90, while talk.religion.misc achieved an F1-score of 0.46.
Saved Model Artifacts
After training, the notebook saves:
best_text_classifier.joblib
tfidf_vectorizer.joblib
These files allow the trained classifier and feature extractor to be reused without retraining.
Project Structure
.
├── nlp_text_classification_pipeline (1).ipynb
├── best_text_classifier.joblib
├── tfidf_vectorizer.joblib
└── README.md
The .joblib files are generated when the notebook reaches the model-saving section.

Installation
Create a Python environment and install the required packages:
pip install scikit-learn pandas numpy nltk matplotlib seaborn joblib
Running the Project
Option 1: Google Colab
Upload the notebook to Google Colab and run the cells sequentially.
The notebook automatically downloads the 20 Newsgroups dataset on its first run.
Option 2: Jupyter Notebook
Clone or download the repository, install the dependencies, and launch Jupyter:
jupyter notebook
Open:
nlp_text_classification_pipeline (1).ipynb
Run the cells from top to bottom.
Inference Example
The notebook includes a reusable prediction function:
def predict_category(raw_text, model=best_model, vec=vectorizer):
    cleaned = clean_text(raw_text)
    vectorized = vec.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction
Example:
text = "The new graphics card significantly improves rendering performance in games."
print(predict_category(text))
The notebook predicts:
comp.graphics
Another example:
text = "NASA announced a new mission to study the moons of Jupiter."
print(predict_category(text))
The notebook predicts:
sci.space
Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook / Google Colab
Key Learning Outcomes
This project demonstrates practical implementation of:
- NLP text cleaning
- Stop-word removal
- Lemmatization
- TF-IDF feature extraction
- N-gram representation
- Multi-class classification
- Naive Bayes
- Logistic Regression
- Linear SVM
- Model comparison
- Classification metrics
- Confusion-matrix analysis
- Model serialization
- Text inference
Extending the Project
The notebook identifies several possible extensions:
Custom Dataset
Replace the 20 Newsgroups data with a custom CSV containing text and label columns.
Deep Learning
Replace the classical ML stage with an architecture such as:
- Embedding + LSTM
- CNN for text classification
Transformers
Fine-tune a pretrained Transformer such as:
distilbert-base-uncased
using Hugging Face transformers and datasets.
Hyperparameter Optimization
Use:
GridSearchCV
RandomizedSearchCV
to optimize the TF-IDF and classifier parameters.
Class Imbalance
For an imbalanced custom dataset, consider class weighting or resampling techniques.
Limitations
- The reported results are based on the 20 Newsgroups benchmark and may not generalize to other domains.
- The pipeline uses classical TF-IDF features rather than contextual embeddings.
- The notebook's sample inference includes an example where a healthcare-related sentence is classified as sci.space; this illustrates that predictions depend on the training data and feature representation and should not be treated as semantic understanding.
- Saved .joblib artifacts should only be loaded from trusted sources.
License
This repository does not specify a license in the supplied notebook. Add an appropriate license before distributing the project publicly.
