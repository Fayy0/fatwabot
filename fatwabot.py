import pandas as pd
import re
from nltk.tokenize import word_tokenize
from farasa.stemmer import FarasaStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
# Install necessary packages
# You can install the required packages in your terminal before running the script
# pip install nltk farasa

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import subprocess

# Install Java
subprocess.run(['sudo', 'apt-get', 'update'])
subprocess.run(['sudo', 'apt-get', 'install', 'default-jre', '-y'])


# Load your Excel file into a DataFrame
file_path = '/workspaces/fatwabot/Al Shaikh Abdual Aziz Al Ashaikh_after removing frequent words.xlsx'
df = pd.read_excel(file_path, sheet_name='data')  # Adjust sheet_name if needed

# Arabic text processing
def preprocess_arabic_text(text):
    text = re.sub(r'[^ء-ي\s]', '', text)
    words = word_tokenize(text)
    words = [re.sub(r'[\u064B-\u0652]', '', word) for word in words]
    stop_words = set(nltk.corpus.stopwords.words('arabic'))
    words = [word for word in words if word not in stop_words]
    stemmer = FarasaStemmer(interactive=True)
    words = [stemmer.stem(word) for word in words]
    processed_text = ' '.join(words)
    return processed_text

# Process 'Column1.question'
df['processed_question'] = df['Column1.question'].apply(preprocess_arabic_text)

# Process 'Column1.answer'
df['processed_answer'] = df['Column1.answer'].apply(preprocess_arabic_text)

# Define the features and target
X = df[['processed_question', 'processed_answer']]
y = df['Column1._class']  # Assuming 'Column1._class' is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 1))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.apply(lambda x: ' '.join(x), axis=1))
X_test_tfidf = tfidf_vectorizer.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

# Model training
# Logistic Regression
logistic_regression = LogisticRegression(verbose=0, max_iter=100, solver='lbfgs', C=1.0, penalty='l2')
logistic_regression.fit(X_train_tfidf, y_train)
y_pred_lr = logistic_regression.predict(X_test_tfidf)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='gini',
                                       min_samples_leaf=1, min_samples_split=2, random_state=None)
random_forest.fit(X_train_tfidf, y_train)
y_pred_rf = random_forest.predict(X_test_tfidf)

# XGBoost
xgb_classifier = XGBClassifier(booster='gbtree', eta=0.3, min_child_weight=1, max_depth=6, scale_pos_weight=1)
xgb_classifier.fit(X_train_tfidf, y_train)
y_pred_xgb = xgb_classifier.predict(X_test_tfidf)

# Multi-layer Perceptron (MLP)
mlp_classifier = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, max_iter=200, shuffle=True, verbose=False)
mlp_classifier.fit(X_train_tfidf, y_train)
y_pred_mlp = mlp_classifier.predict(X_test_tfidf)

# Ensemble Classifier
ensemble_classifier = VotingClassifier(estimators=[
    ('lr', logistic_regression),
    ('rf', random_forest),
    ('xgb', xgb_classifier),
    ('mlp', mlp_classifier)
], voting='hard')

ensemble_classifier.fit(X_train_tfidf, y_train)
y_pred_ensemble = ensemble_classifier.predict(X_test_tfidf)

# Evaluate performance
def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Evaluate each model
accuracy_lr, precision_lr, recall_lr, f1_lr = evaluate_performance(y_test, y_pred_lr)
accuracy_rf, precision_rf, recall_rf, f1_rf = evaluate_performance(y_test, y_pred_rf)
accuracy_xgb, precision_xgb, recall_xgb, f1_xgb = evaluate_performance(y_test, y_pred_xgb)
accuracy_mlp, precision_mlp, recall_mlp, f1_mlp = evaluate_performance(y_test, y_pred_mlp)
accuracy_ensemble, precision_ensemble, recall_ensemble, f1_ensemble = evaluate_performance(y_test, y_pred_ensemble)



# After training the model and having the tfidf_vectorizer object
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# # Save TF-IDF vectorizer
# with open('tfidf_vectorizer.joblib', 'wb') as file:
#     joblib.dump(tfidf_vectorizer, file)

# # Save Random Forest model
# with open('random_forest_model.joblib', 'wb') as file:
#     joblib.dump(random_forest_model, file)


# Print evaluation results
print("Logistic Regression:")
print(f"Accuracy: {accuracy_lr}, Precision: {precision_lr}, Recall: {recall_lr}, F1 Score: {f1_lr}")

print("\nRandom Forest:")
print(f"Accuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}, F1 Score: {f1_rf}")

print("\nXGBoost:")
print(f"Accuracy: {accuracy_xgb}, Precision: {precision_xgb}, Recall: {recall_xgb}, F1 Score: {f1_xgb}")

print("\nMulti-layer Perceptron:")
print(f"Accuracy: {accuracy_mlp}, Precision: {precision_mlp}, Recall: {recall_mlp}, F1 Score: {f1_mlp}")

print("\nEnsemble Classifier:")
print(f"Accuracy: {accuracy_ensemble}, Precision: {precision_ensemble}, Recall: {recall_ensemble}, F1 Score: {f1_ensemble}")
