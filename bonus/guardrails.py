import re

def validate_answer(question: str, answer: str) -> bool:
    if not answer or not answer.strip():
        return False
    numeric_keywords = ["revenue", "income", "profit", "billion", "million",
                        "thousand", "percent", "growth", "increase"]
    if any(kw in question.lower() for kw in numeric_keywords):
        if not re.search(r"\d", answer):
            return False
    return True

def safe_fallback_answer() -> str:
    return "I'm sorry, but I couldn't provide a valid answer. Please rephrase your question or try again later."