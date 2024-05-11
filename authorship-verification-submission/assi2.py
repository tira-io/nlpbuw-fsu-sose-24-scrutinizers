import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import langid

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    # Predict language for each text snippet using langid
    predictions = []
    for idx, example in tqdm(text_validation.iterrows(), total=len(text_validation)):
        lang, _ = langid.classify(example['text'])
        predictions.append({"id": example['id'], "lang": lang})

    # Save predictions to file
    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + '\n')
