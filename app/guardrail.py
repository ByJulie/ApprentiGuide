"""Input guardrail, run on every question before anything else (pipeline.answer and api.py).

Three rules written in code, so they hold whatever the model does:
1. Personal data (email, phone number, social security number, IBAN) is masked before the question
   leaves the laptop: the rewrite and the answer are written by a cloud model (Groq). Nothing is logged.
2. A question that tries to change the assistant's rules (prompt injection) gets a fixed refusal.
3. A message of distress gets a fixed message with help lines, never a legal answer.
Off-topic questions are not filtered here: retrieval finds no text about them and the abstention
threshold answers NOT FOUND (measured on the "no answer" family of the test set).
"""
import re
from dataclasses import dataclass, field

PII = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b"),
    # French social security number: sex, year, month, department (2A/2B for Corsica), commune, order, key
    "nir": re.compile(r"\b[12][\s.]?\d{2}[\s.]?\d{2}[\s.]?(?:\d{2}|2A|2B)[\s.]?\d{3}[\s.]?\d{3}(?:[\s.]?\d{2})?\b"),
    "phone": re.compile(r"(?<!\d)(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)"),
}

INJECTION = re.compile(
    r"ignore[sz]?\s+(?:all\s+|toutes?\s+|the\s+|les\s+)?(?:your\s+|tes\s+|vos\s+)?(?:previous\s+|précédentes?\s+)?"
    r"(?:instructions|rules|règles|consignes|sources)"
    r"|(?:system\s+prompt|prompt\s+syst[eè]me|ton\s+prompt|your\s+prompt)"
    r"|^\s*system\s*:|nouvelle\s+r[eè]gle|new\s+rule"
    r"|(?:tu\s+es|you\s+are)\s+(?:maintenant|now)\b"
    r"|answer\s+from\s+your\s+own\s+knowledge|sans\s+citer\s+de\s+source", re.I | re.M)

DISTRESS = re.compile(
    r"\b(?:suicid\w*|en\s+finir|me\s+tuer|mourir|plus\s+envie\s+de\s+vivre|kill\s+myself|end\s+my\s+life|want\s+to\s+die)\b", re.I)

ENGLISH = re.compile(r"\b(?:the|is|are|can|my|what|how|do|does|i|you|should|work|with)\b", re.I)
FRENCH = re.compile(r"\b(?:le|la|les|est|je|mon|ma|mes|un|une|des|que|quoi|peux|puis|pour|avec)\b", re.I)

MESSAGES = {
    "injection": {
        "fr": ("Je réponds uniquement aux questions sur l'apprentissage, le travail et le séjour des étudiants, "
               "à partir des textes officiels, et je ne peux pas changer ces règles. Pose ta question sans consigne "
               "et je chercherai la réponse dans les textes."),
        "en": ("I only answer questions about apprenticeship, work and residence rights of students, from the "
               "official texts, and I cannot change these rules. Ask your question without instructions and I "
               "will look for the answer in the texts.")},
    "distress": {
        "fr": ("Je suis désolé que tu traverses ça. Tu n'es pas seul·e : appelle le 3114 (numéro national de "
               "prévention du suicide, gratuit, 24 h/24) ou le 15 en cas d'urgence. Tu peux aussi en parler au "
               "service santé ou au référent handicap et bien-être de ton école, ou à ton maître d'apprentissage. "
               "Si ton alternance se passe mal, l'école et le CFA peuvent t'aider, y compris pour une médiation."),
        "en": ("I am sorry you are going through this. You are not alone: call 3114 (French national suicide "
               "prevention line, free, 24/7) or 15 in an emergency. You can also talk to your school's health "
               "or student support service, or to your apprenticeship master. If your apprenticeship is going "
               "badly, the school and the CFA can help, including with mediation.")},
}


@dataclass
class Check:
    question: str                      # the question with personal data masked: the only one sent further
    lang: str
    blocked: str | None = None         # "injection" or "distress": answer with message, no retrieval
    pii: list[str] = field(default_factory=list)

    @property
    def message(self) -> str | None:
        return MESSAGES[self.blocked][self.lang] if self.blocked else None


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Text with personal data replaced by [email], [phone]...; and the kinds found (never the values)."""
    kinds = []
    for kind, pattern in PII.items():
        text, n = pattern.subn(f"[{kind}]", text)
        if n:
            kinds.append(kind)
    return text, kinds


def guess_lang(text: str) -> str:
    return "en" if len(ENGLISH.findall(text)) > len(FRENCH.findall(text)) else "fr"


def check(question: str) -> Check:
    masked, kinds = mask_pii(question)
    result = Check(question=masked, lang=guess_lang(masked), pii=kinds)
    if DISTRESS.search(masked):
        result.blocked = "distress"
    elif INJECTION.search(masked):
        result.blocked = "injection"
    return result
