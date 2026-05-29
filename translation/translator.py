from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"
    
def translate_to_english(text: str) -> tuple[str, str]:
    lang = detect_language(text)

    if lang == "en":
        return text, "en"
    translated = GoogleTranslator(source=lang, target="en").translate(text)
    return translated, lang
    
def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    return GoogleTranslator(source="en", target=target_lang).translate(text)
        
    