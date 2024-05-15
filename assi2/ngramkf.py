from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    tira = Client()

    # loading validation data 
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    # Initialize CountVectorizer with n-grams
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))  # Using bi-grams

    # Fit CountVectorizer on the text data
    ngram_matrix = ngram_vectorizer.fit_transform(text_validation['text'])

    # Train a simple KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(ngram_matrix, targets_validation['lang'])

    # Predict languages for validation set
    predicted_languages = knn_classifier.predict(ngram_vectorizer.transform(text_validation['text']))

    # Save predictions to JSON lines file
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions_file = Path(output_directory) / "predictions.jsonl"
    predictions_df = pd.DataFrame({
        "id": text_validation["id"],
        "lang": predicted_languages
    })
    predictions_df.to_json(predictions_file, orient="records", lines=True)

    print(f"Predictions saved to {predictions_file}")
