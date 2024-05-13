import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Heuristics for language identification
def identify_language(text):
    # List of common words/phrases in different languages
    language_patterns = {
    'en': ['the', 'and', 'of', 'to', 'a', 'in', 'is'],
    'es': ['de', 'la', 'que', 'el', 'en', 'y', 'a'],
    'fr': ['de', 'le', 'et', 'à', 'la', 'les', 'des'],
    'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu'],
    'pl': ['w', 'i', 'z', 'na', 'do', 'że', 'jest'],
    'zh': ['的', '了', '是', '在', '我', '有', '他'],
    'no': ['og', 'i', 'på', 'til', 'det', 'er', 'en'],
    'hr': ['i', 'je', 'na', 'u', 'za', 'od', 'da'],
    'ko': ['의', '에', '을', '이', '는', '가', '들'],
    'af': ['die', 'van', 'en', 'in', 'is', 'dat', 'vir'],
    'fi': ['ja', 'on', 'että', 'ei', 'olen', 'mutta', 'ole'],
    'it': ['di', 'che', 'è', 'non', 'la', 'il', 'a'],
    'ur': ['کے', 'کا', 'میں', 'کی', 'ہے', 'کو', 'نہیں'],
    'ru': ['и', 'в', 'на', 'с', 'что', 'как', 'не'],
    'el': ['και', 'το', 'της', 'του', 'την', 'στο', 'με'],
    'nl': ['van', 'de', 'het', 'een', 'dat', 'is', 'in'],
    'az': ['və', 'ki', 'bir', 'ə', 'bu', 'da', 'il'],
    'da': ['det', 'er', 'en', 'jeg', 'du', 'har', 'af'],
    'bg': ['на', 'да', 'се', 'не', 'ще', 'съм', 'аз'],
    'cs': ['je', 'v', 'se', 'na', 'že', 'pro', 'do'],
    'sv': ['och', 'att', 'det', 'är', 'en', 'jag', 'inte']
}

    
    # Calculate the frequency of occurrence of language-specific patterns
    lang_scores = {lang: sum(text.lower().count(word) for word in patterns) for lang, patterns in language_patterns.items()}
    
    # Return the language with the highest score
    return max(lang_scores, key=lang_scores.get)

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    # Predict language for each text snippet using heuristics
    predictions = []
    for idx, example in tqdm(text_validation.iterrows(), total=len(text_validation)):
        lang = identify_language(example['text'])
        predictions.append({"id": example['id'], "lang": lang})

    # Save predictions to file
    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + '\n')
