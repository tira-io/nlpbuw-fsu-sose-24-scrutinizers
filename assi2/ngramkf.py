from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit TF-IDF vectorizer on the text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_validation['text'])

    # Train a simple KNN classifier on the TF-IDF vectors
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(tfidf_matrix, targets_validation['lang'])

    # Predict languages for the validation set
    predicted_languages = knn_classifier.predict(tfidf_vectorizer.transform(text_validation['text']))

    # Save predictions to JSON lines file
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions_file = Path(output_directory) / "predictions.jsonl"
    predictions_df = pd.DataFrame({
        "id": text_validation["id"],
        "lang": predicted_languages
    })
    predictions_df.to_json(predictions_file, orient="records", lines=True)

    print(f"Predictions saved to {predictions_file}")
