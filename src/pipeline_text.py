from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import re
from collections import defaultdict

from .kb import load_kb, build_passages
from .alias import build_alias_map, normalize_query
from .language import detect_lang
from .retrieval import load_index, load_meta, retrieve
from .llm_local import generate
from .translate import translate_text
from .cleaning import clean_output


def refusal_message(lang: str) -> str:
    msgs = {
        "en": "I don't know based on my sources. Please ask a human guide.",
        "ar": "لا أعرف بناءً على مصادري. من فضلك اسأل مرشدًا بشريًا.",
        "de": "Ich weiß es anhand meiner Quellen nicht. Bitte fragen Sie einen menschlichen Guide.",
        "fr": "Je ne sais pas d’après mes sources. Veuillez demander à un guide humain.",
        "it": "Non lo so in base alle mie fonti. Chiedi a una guida umana, per favore.",
        "es": "No lo sé según mis fuentes. Por favor pregunta a un guía humano.",
        "zh": "我无法根据现有资料确定答案。请咨询人工导览员。",
        "ru": "Я не уверен(а) на основе доступных источников. Пожалуйста, спросите у человека-гида.",
        "pt": "Não tenho certeza com base nas minhas fontes. Por favor, pergunte a um guia humano.",
        "tr": "Mevcut kaynaklarıma dayanarak emin değilim. Lütfen bir insan rehbere sorun.",
        "ja": "手元の情報だけでは確実に答えられません。人間のガイドにお尋ねください。",
    }
    return msgs.get(lang, msgs["en"])


def duplicate_title_message(lang: str, title: str) -> str:
    msgs = {
        "en": f'I found multiple artifacts with the title "{title}". Which one do you mean?',
        "ar": f'وجدت عدة قطع أثرية بعنوان "{title}". أي واحدة تقصد؟',
        "de": f'Ich habe mehrere Artefakte mit dem Titel "{title}" gefunden. Welches meinen Sie?',
        "fr": f'J’ai trouvé plusieurs artefacts portant le titre "{title}". Lequel voulez-vous dire ?',
        "it": f'Ho trovato più manufatti con il titolo "{title}". Quale intendi?',
        "es": f'Encontré varios artefactos con el título "{title}". ¿Cuál quieres decir?',
        "zh": f'我找到了多个标题为“{title}”的文物。你指的是哪一个？',
        "ru": f'Я нашёл несколько артефтов с названием "{title}". Какой именно вы имеете в виду?',
        "pt": f'Encontrei vários artefatos com o título "{title}". Qual deles você quer dizer?',
        "tr": f'"{title}" başlıklı birden fazla eser buldum. Hangisini kastediyorsunuz?',
        "ja": f'「{title}」というタイトルの展示物が複数見つかりました。どれを指していますか？',
    }
    return msgs.get(lang, msgs["en"])


def build_prompt_english(question: str, retrieved) -> str:
    context = "\n\n".join([f"[{i+1}] {r['passage']}" for i, r in enumerate(retrieved)])
    return f"""
You are a museum guide assistant.

Use ONLY the CONTEXT below. The context is museum catalog metadata.
Write a helpful answer by summarizing what the object is using those fields (title, object type, culture, date, medium, dimensions, repository, tags, link).
Do NOT refuse. Do NOT say you don't know. Do NOT ask for a human guide.
Do NOT add facts that are not in the context.

Answer ONLY in English. Keep it to 2 to 5 sentences.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()


_STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "it", "as", "about",
    "what", "when", "where", "who", "how", "tell", "me", "please", "explain", "describe"
}


def _tokenize(text: str):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS_EN and len(t) > 2]


def lexical_overlap_count(question: str, passage: str) -> int:
    q_tokens = set(_tokenize(question))
    p_tokens = set(_tokenize(passage))
    return len(q_tokens.intersection(p_tokens))


def build_artifact_option_label(artifact: dict) -> str:
    text = (artifact.get("text_en") or artifact.get("text") or "").strip()

    culture = ""
    date = ""
    material = ""

    m_culture = re.search(r"Culture:\s*([^\.]+)", text, flags=re.IGNORECASE)
    if m_culture:
        culture = m_culture.group(1).strip()

    m_date = re.search(r"Date:\s*([^\.]+)", text, flags=re.IGNORECASE)
    if m_date:
        date = m_date.group(1).strip()

    m_material = re.search(r"Material/Medium:\s*([^\.]+)", text, flags=re.IGNORECASE)
    if m_material:
        material = m_material.group(1).strip()

    parts = [p for p in [culture, date, material] if p]
    if parts:
        return " — " + ", ".join(parts[:3])

    return ""


class TextChatbot:
    def __init__(
        self,
        project_root: str | Path,
        kb_path: str | Path = "data/kb/artifacts_kb.json",
        index_path: str | Path = "data/indexes/faiss.index",
        meta_path: str | Path = "data/indexes/meta.json",
        llm_path: str | Path = "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf"
    ):
        self.project_root = Path(project_root)

        self.kb_path = self.project_root / kb_path
        self.index_path = self.project_root / index_path
        self.meta_path = self.project_root / meta_path
        self.llm_path = self.project_root / llm_path

        artifacts = load_kb(self.kb_path)
        self.artifacts = artifacts
        self.passages, _pass_meta = build_passages(artifacts)
        self.alias_map = build_alias_map(artifacts)

        self.title_to_artifacts = defaultdict(list)
        for a in artifacts:
            title = (a.get("title_en") or a.get("title") or "").strip()
            if title:
                self.title_to_artifacts[title].append(a)

        self.index = load_index(self.index_path)
        self.meta = load_meta(self.meta_path)

    def ask(self, user_question: str, top_k: int = 3) -> Dict[str, Any]:
        m = re.match(r"^\[LANG=([a-z]{2})\]\s*(.*)$", user_question.strip(), flags=re.IGNORECASE)
        forced_lang = None
        if m:
            forced_lang = m.group(1).lower()
            user_question = m.group(2)

        user_lang = forced_lang if forced_lang else detect_lang(user_question)

        norm_q = normalize_query(user_question, self.alias_map)

        query_for_retrieval = norm_q
        if user_lang != "en" and user_lang != "unknown":
            query_for_retrieval = translate_text(norm_q, "en")

        initial_results = retrieve(
            self.index,
            self.meta,
            self.passages,
            query_for_retrieval,
            top_k=10
        )

        retrieved = sorted(
            initial_results,
            key=lambda r: (
                lexical_overlap_count(query_for_retrieval, r["passage"]),
                r["score"]
            ),
            reverse=True
        )[:top_k]

        if retrieved:
            top_title = retrieved[0]["meta"]["title"]
            same_title_artifacts = self.title_to_artifacts.get(top_title, [])

            if len(same_title_artifacts) > 1:
                options = []
                for a in same_title_artifacts[:5]:
                    title = a.get("title_en") or a.get("title") or ""
                    label = f"{title}{build_artifact_option_label(a)}"
                    options.append({
                        "id": a.get("id"),
                        "title": title,
                        "label": label,
                        "text_preview": (a.get("text_en") or a.get("text") or "")[:160]
                    })

                return {
                    "lang": user_lang,
                    "top_score": retrieved[0]["score"],
                    "normalized_question": norm_q,
                    "retrieval_query": query_for_retrieval,
                    "answer": duplicate_title_message(user_lang, top_title),
                    "needs_clarification": True,
                    "clarification_type": "duplicate_title",
                    "duplicate_title": top_title,
                    "options": options,
                    "sources": [r["meta"]["title"] for r in retrieved],
                }

        top_passage = retrieved[0]["passage"] if retrieved else ""
        overlap = lexical_overlap_count(query_for_retrieval, top_passage)

        MIN_OVERLAP = 2
        VERY_HIGH_SCORE = 0.80
        top_score = retrieved[0]["score"] if retrieved else 0.0

        if (not retrieved) or ((top_score < VERY_HIGH_SCORE) and (overlap < MIN_OVERLAP)):
            return {
                "lang": user_lang,
                "top_score": top_score,
                "normalized_question": norm_q,
                "retrieval_query": query_for_retrieval,
                "answer": refusal_message(user_lang),
                "sources": [r["meta"]["title"] for r in retrieved],
            }

        score_threshold = 0.72
        if top_score < score_threshold:
            return {
                "lang": user_lang,
                "top_score": top_score,
                "normalized_question": norm_q,
                "retrieval_query": query_for_retrieval,
                "answer": refusal_message(user_lang),
                "sources": [r["meta"]["title"] for r in retrieved],
            }

        prompt_en = build_prompt_english(norm_q, retrieved)
        answer_en = generate(prompt_en, max_tokens=220)
        answer_en = clean_output(answer_en)

        if "i don't know" in answer_en.lower() or "ask a human guide" in answer_en.lower():
            prompt_en_2 = build_prompt_english(
                "Summarize the object using the metadata fields from CONTEXT (title, object type, culture, date, medium, dimensions, repository, tags, link).",
                retrieved
            )
            answer_en = generate(prompt_en_2, max_tokens=220)
            answer_en = clean_output(answer_en)

        if user_lang == "en":
            return {
                "lang": user_lang,
                "top_score": top_score,
                "normalized_question": norm_q,
                "retrieval_query": query_for_retrieval,
                "answer": answer_en,
                "sources": [r["meta"]["title"] for r in retrieved],
            }

        translated = translate_text(answer_en, user_lang)
        translated = clean_output(translated)

        return {
            "lang": user_lang,
            "top_score": top_score,
            "normalized_question": norm_q,
            "retrieval_query": query_for_retrieval,
            "answer": translated,
            "answer_en": answer_en,
            "sources": [r["meta"]["title"] for r in retrieved],
        }