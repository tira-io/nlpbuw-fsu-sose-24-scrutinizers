from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"language-identification-validation-20240429-training"
    )

    # Load the model and make predictions
    vectorizer = load(Path(__file__).parent / "vectorizer.joblib")
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(vectorizer.transform(df['text']))
    df["lang"] = predictions
    df = df[["id", "lang"]]


    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
