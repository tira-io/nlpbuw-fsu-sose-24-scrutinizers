from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from joblib import dump

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    print('TF-IDF vectorizer initialized')
    # Fit TF-IDF vectorizer on the text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_validation['text'])
    # save the vectorizer
    dump(tfidf_vectorizer, Path(__file__).parent / "vectorizer.joblib")
    # Train a simple KNN classifier on the TF-IDF vectors
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(tfidf_matrix, targets_validation['lang'])
    # save the model
    dump(knn_classifier, Path(__file__).parent / "model.joblib")
