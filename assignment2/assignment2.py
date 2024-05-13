import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def identify_language(text):
    # List of common words/phrases in different languages
    language_patterns = {
    'en': ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'with', 'as', 'I', 'his', 'they', 'be', 'at', 'one', 'have', 'this', 'from', 'or', 'had', 'by', 'hot', 'word', 'but', 'what', 'some', 'we', 'can', 'out', 'other', 'were', 'all', 'there', 'when', 'up', 'use', 'your', 'how', 'said', 'an'],
    'es': ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'del', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno'],
    'fr': ['de', 'le', 'et', 'à', 'la', 'les', 'des', 'un', 'est', 'que', 'pour', 'qui', 'dans', 'sur', 'plus', 'se', 'ce', 'ne', 'pas', 'par', 'au', 'son', 'avec', 'ils', 'comme', 'leurs', 'mais', 'elle', 'ses', 'aussi', 'sous', 'ou', 'alors', 'après', 'être', 'avoir', 'fait', 'deux', 'bien', 'où', 'dont', 'même', 'pendant', 'aussi', 'contre', 'jamais', 'nos', 'dire', 'si'],
    'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'Die', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass', 'sie', 'nach', 'wurde', 'wird', 'man', 'bei', 'einer', 'wir', 'das', 'sind', 'von', 'hatte', 'kann', 'aber', 'nichts', 'sind', 'so', 'da', 'oder', 'wie', 'Jetzt', 'ihr', 'wo'],
    'pl': ['w', 'i', 'z', 'na', 'do', 'że', 'jest', 'z', 'za', 'nie', 'o', 'jak', 'po', 'czy', 'aby', 'przez', 'się', 'to', 'już', 'lub', 'co', 'ale', 'bardzo', 'więcej', 'tylko', 'który', 'tego', 'bo', 'może', 'wszystko', 'gdy', 'tak', 'teraz', 'jestem', 'ci', 'nim', 'która', 'tam', 'je', 'nas', 'ja', 'więcej', 'było', 'ale', 'coś', 'jego', 'kiedy', 'ty', 'mnie', 'jego'],
    'zh': ['的', '了', '是', '在', '我', '有', '他', '不', '人', '这', '个', '上', '们', '来', '到', '时', '大', '地', '为', '子', '中', '你', '说', '生', '国', '年', '着', '就', '那', '和', '要', '她', '出', '也', '得', '里', '后', '自', '以', '会', '家', '可', '下', '而', '过', '天', '去', '能', '对', '小', '多', '然', '于', '心'],
    'no': ['og', 'i', 'på', 'til', 'det', 'er', 'en', 'av', 'for', 'å', 'med', 'som', 'den', 'har', 'de', 'ikke', 'at', 'vi', 'hadde', 'det', 'fra', 'kan', 'du', 'vil', 'da', 'var', 'meg', 'seg', 'eller', 'oss', 'hun', 'ved', 'et', 'om', 'man', 'min', 'noen', 'så', 'ble', 'hva', 'skal', 'men', 'over', 'etter', 'bare', 'hvor', 'etter', 'også', 'selv', 'mot', 'mot', 'hvis'],
    'hr': ['i', 'je', 'na', 'u', 'za', 'od', 'da', 'a', 'ali', 'bi', 'će', 'što', 'su', 'sa', 'koji', 'iz', 'ili', 'ako', 'sve', 'njega', 'smo', 'ovo', 'kao', 'do', 'ova', 'zašto', 'ovaj', 'koje', 'onda', 'ona', 'ovu', 'sam', 'mi', 'još', 'biti', 'bilo', 'koja', 'ova', 'ali', 'to', 'samo', 'kako', 'ne', 'čak', 'nije', 'jer', 'nego', 'me', 'jer', 'ti', 'šta'],
    'ko': ['의', '에', '을', '이', '는', '가', '들', '그', '와', '로', '하', '것', '사람', '들', '에게', '있는', '있다', '개', '그들', '이다', '한', '그리고', '되는', '수', '보다', '년', '다', '시간', '그들의', '일', '이러한', '우리', '할', '수있다', '그것은', '없는', '우리는', '우리의', '일본', '일본어', '자신의', '또한', '더', '자신', '자신을', '있었다', '좋은', '아니라', '우리가', '당신이', '더', '우리를', '우리가'],
    'af': ['die', 'van', 'en', 'in', 'is', 'dat', 'vir', 'op', 'met', 'om', 'te', 'die', 'nie', 'hy', 'wat', 'n', 'maar', 'jy', 'my', 'ons', 'hulle', 'hulle', 'waar', 'kyk', 'kan', 'na', 'sien', 'sal', 'as', 'een', 'het', 'of', 'hoe', 'ek', 'nie', 'het', 'aan', 'ook', 'so', 'wat', 'jou', 's', 'uit', 'was', 'ons', 'het', 'nie', 'haar', 'nie', 'hy', 'o', 'kan'],
    'fi': ['ja', 'on', 'että', 'ei', 'olen', 'mutta', 'ole', 'hän', 'me', 'mitä', 'siinä', 'hänen', 'ovat', 'tai', 'sillä', 'heidän', 'tässä', 'täytyy', 'niin', 'tämä', 'kuin', 'olet', 'niitä', 'tämän', 'tiedän', 'enää', 'häntä', 'meidän', 'sitä', 'jo', 'en', 'vain', 'pitää', 'tai', 'aina', 'tällä', 'sinun', 'miksi', 'mutta', 'joka', 'jos', 'heidät', 'hyvin', 'olen', 'ensimmäinen', 'näin', 'kuinka', 'he', 'tiedä'],
    'it': ['di', 'che', 'è', 'non', 'la', 'il', 'a', 'per', 'sono', 'con', 'come', 'io', 'questo', 'da', 'si', 'ma', 'loro', 'suo', 'anzi', 'o', 'così', 'anche', 'avere', 'sono', 'stato', 'mai', 'quando', 'dove', 'lì', 'fare', 'uno', 'lui', 'noi', 'da', 'se', 'fa', 'essa', 'o', 'ma', 'perché', 'chi', 'perché', 'cosa', 'loro', 'vorrei', 'voglio', 'come', 'molto', 'così', 'lei', 'lui'],
    'ur': ['کے', 'کا', 'میں', 'کی', 'ہے', 'کو', 'نہیں', 'کیا', 'کر', 'رہے', 'اور', 'ہو', 'یہ', 'اس', 'اپنے', 'پر', 'ہیں', 'کرنے', 'کہ', 'کیے', 'گیا', 'تھا', 'یہاں', 'وہ', 'کوئی', 'جب', 'ایک', 'نے', 'سب', 'کیونکہ', 'کچھ', 'کبھی', 'اب', 'دوبارہ', 'تو', 'اگر', 'اپنی', 'ہوتا', 'آپ', 'ہوتے', 'جو', 'کون', 'اسے', 'وہاں', 'کہیں', 'یا', 'ہوتی', 'کرتا', 'کیسے', 'جبکہ', 'کرتے'],
    'ru': ['и', 'в', 'на', 'с', 'что', 'как', 'не', 'я', 'это', 'он', 'а', 'ты', 'вы', 'к', 'у', 'так', 'же', 'сказал', 'за', 'они', 'этот', 'о', 'из', 'все', 'был', 'мы', 'может', 'но', 'быть', 'она', 'его', 'когда', 'один', 'где', 'кто', 'время', 'если', 'себя', 'нет', 'сейчас', 'мне', 'есть', 'чем', 'говорит', 'даже', 'только', 'хорошо', 'мой', 'вот', 'который', 'есть', 'хотите'],
    'el': ['και', 'το', 'της', 'του', 'την', 'στο', 'με', 'μια', 'είναι', 'για', 'από', 'ότι', 'ο', 'μας', 'μετά', 'αυτό', 'που', 'αυτή', 'αλλά', 'θα', 'στην', 'ήταν', 'ένα', 'εγώ', 'στον', 'τον', 'ενώ', 'στη', 'έναν', 'σαν', 'όχι', 'μου', 'όλα', 'είχε', 'πολύ', 'τους', 'μπορεί', 'πρέπει', 'αυτήν', 'τόσο', 'της', 'πριν', 'μέσα', 'όπως', 'μόνο', 'σας', 'τις', 'τα', 'ήταν', 'θα', 'όπου', 'μπορείτε'],
    'nl': ['van', 'de', 'het', 'een', 'dat', 'is', 'in', 'je', 'niet', 'hij', 'zijn', 'op', 'aan', 'met', 'voor', 'ik', 'als', 'maar', 'mijn', 'was', 'ik', 'ben', 'aan', 'zo', 'ze', 'om', 'hem', 'dan', 'zou', 'wat', 'nu', 'ge', 'toen', 'niets', 'meer', 'is', 'die', 'u', 'heeft', 'dit', 'kan', 'worden', 'er', 'nog', 'willen', 'gaan', 'over', 'mensen', 'dit', 'zo'],
    'az': ['və', 'ki', 'bir', 'ə', 'bu', 'da', 'il', 'için', 'onun', 'oldu', 'kişi', 'lər', 'var', 'mən', 'ilə', 'gəlir', 'da', 'e', 'istəyirəm', 'istəyirəm', 'görə', 'daha', 'onlar', 'bu', 'harada', 'bilər', 'dəyişən', 'mənə', 'görə', 'hər', 'deyir', 'istifadə', 'edə', 'mümkündür', 'deyir', 'olmaq', 'ona', 'qədər', 'öz', 'mən', 'və', 'daha', 'əgər', 'siz', 'onun', 'onların', 'bu', 'onda', 'edir', 'istifadə'],
    'da': ['det', 'er', 'en', 'jeg', 'du', 'har', 'af', 'for', 'på', 'at', 'han', 'med', 'de', 'den', 'om', 'sig', 'et', 'men', 'vi', 'var', 'min', 'hun', 'nu', 'ved', 'fra', 'du', 'eller', 'bare', 'her', 'op', 'vil', 'godt', 'os', 'have', 'hende', 'alle', 'over', 'når', 'alt', 'blev', 'deres', 'mig', 'jo', 'mod', 'disse', 'nogle', 'ser', 'mere', 'ud', 'kom', 'hvad'],
    'bg': ['на', 'да', 'се', 'не', 'ще', 'съм', 'аз', 'то', 'че', 'те', 'той', 'както', 'тя', 'все', 'щом', 'от', 'е', 'или', 'ама', 'при', 'но', 'ме', 'беше', 'ще', 'която', 'ако', 'този', 'вече', 'беше', 'до', 'него', 'бях', 'нещо', 'по', 'където', 'всичко', 'пък', 'са', 'ли', 'също', 'ни', 'ми', 'защо', 'само', 'пак', 'им', 'няма', 'докато', 'даже', 'ви'],
    'cs': ['je', 'v', 'se', 'na', 'že', 'pro', 'do', 'co', 'jak', 'jsem', 'ale', 'k', 'si', 'nebo', 'tak', 'to', 'jen', 'mě', 'než', 'jako', 'teď', 'jsem', 'ani', 'byl', 'asi', 'budu', 'mu', 'tady', 'jen', 'pak', 'už', 'také', 'by', 'jenom', 'může', 'že', 'než', 've', 'tom', 'ale', 'proto', 'nechce', 'když', 'jsem', 'já', 'kdy', 'jsou', 'své', 'dost', 'jste', 'vám'],
    'sv': ['och', 'att', 'det', 'är', 'en', 'jag', 'inte', 'för', 'han', 'med', 'på', 'sig', 'som', 'att', 'han', 'för', 'med', 'inte', 'hans', 'hon', 'om', 'vara', 'har', 'en', 'det', 'att', 'honom', 'så', 'hon', 'ett', 'men', 'det', 'som', 'på', 'hon', 'vara', 'vi', 'med', 'de', 'kommer', 'han', 'av', 'dem', 'om', 'det', 'säger', 'du', 'min', 'vara', 'på', 'sina']
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
