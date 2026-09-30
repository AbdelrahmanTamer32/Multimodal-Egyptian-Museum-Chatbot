from __future__ import annotations
from .llm_local import generate
import re


_LANG_INSTR = {
    "en": "Translate to English.",
    "ar": "ترجم إلى العربية فقط.",
    "de": "Übersetze nur ins Deutsche.",
    "fr": "Traduis uniquement en français.",
    "it": "Traduci solo in italiano.",
    "es": "Traduce solo al español.",
    "zh": "Translate to Simplified Chinese.",
    "ru": "Translate to Russian.",
    "pt": "Translate to Portuguese.",
    "tr": "Translate to Turkish.",
    "ja": "Translate to Japanese.",
    "ko": "Translate to Korean.",
    
}


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _contains_too_much_latin(text: str) -> bool:
    latin_count = len(re.findall(r"[A-Za-z]", text))
    return latin_count >= 6


def _contains_non_arabic(text: str) -> bool:
    return _contains_cjk(text) or _contains_too_much_latin(text)


def _too_short(text: str, min_chars: int = 120) -> bool:
    return len((text or "").strip()) < min_chars


def _sanitize_arabic(text: str) -> str:
    """
    Hard-remove any CJK + Latin letters from Arabic output.
    Keeps Arabic letters, digits, punctuation, and spaces.
    """
    if not text:
        return text

    # Remove CJK characters
    text = re.sub(r"[\u4e00-\u9fff]+", "", text)

    # Remove Latin words (and leftover Latin letters)
    text = re.sub(r"[A-Za-z]+", "", text)

    # Clean extra spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def translate_text(text: str, target_lang: str) -> str:
    # --- Arabic special case: Arabic-only output + validate + retry + sanitize ---
    if target_lang == "ar":
        prompt = f"""
ترجم النص التالي إلى العربية الفصحى فقط.

قواعد صارمة:
- اكتب بالعربية فقط (بدون أي كلمات إنجليزية أو صينية).
- لا تضف شرحًا خارج معنى النص.
- حافظ على المعنى نفسه.
- اكتب إجابة كاملة وواضحة في فقرة واحدة.

النص:
{text}

الإجابة العربية:
""".strip()

        out = generate(prompt, max_tokens=380).strip()

        # Retry if contaminated
        if _contains_non_arabic(out):
            prompt2 = f"""
اكتب الإجابة بالعربية فقط (ممنوع أي حروف غير عربية، وممنوع أي كلمات إنجليزية أو صينية).
اكتب فقرة واحدة كاملة وواضحة.
إذا لم تستطع، اكتب: لا أعرف بناءً على مصادري.

النص:
{text}

الإجابة العربية:
""".strip()
            out = generate(prompt2, max_tokens=380).strip()

        # Retry if too short
        if _too_short(out):
            prompt3 = f"""
ترجم النص التالي إلى العربية الفصحى ترجمة كاملة دون تلخيص.

قواعد صارمة:
- اكتب بالعربية فقط.
- لا تلخص ولا تختصر: انقل كل التفاصيل الموجودة في النص.
- لا تضف معلومات جديدة.
- اكتب 2 إلى 4 جمل واضحة وكاملة.

النص:
{text}

الترجمة العربية الكاملة:
""".strip()
            out = generate(prompt3, max_tokens=420).strip()

        # Final hard sanitize (guarantees no Chinese/Latin)
        out = _sanitize_arabic(out)

        # If sanitize made it too short, fallback to a safe refusal in Arabic
        if _too_short(out, min_chars=40):
            out = "لا أعرف بناءً على مصادري. من فضلك اسأل مرشدًا بشريًا."

        return out

    # --- Other languages: keep normal translation ---
    instr = _LANG_INSTR.get(target_lang, "Translate to the target language.")

    prompt = f"""
You are a professional translator.
{instr}

STRICT RULES:
- Output ONLY in the target language.
- Do NOT mix languages.
- Do NOT add explanations, notes, or extra sentences.
- Keep the meaning identical.
- Output ONLY the translation text (no quotes, no markdown).

TEXT:
{text}

TRANSLATION:
""".strip()
    return generate(prompt, max_tokens=220).strip()