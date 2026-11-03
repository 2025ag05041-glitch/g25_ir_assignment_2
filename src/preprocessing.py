"""Configurable text preprocessing pipeline.

Order of steps: noise cleaning -> lowercasing -> tokenization -> stop-word
removal -> stemming or lemmatization -> n-gram generation.
NLTK is used only for the stop-word list, the Porter stemmer and the WordNet
lemmatizer. Built-in fallbacks are used if the NLTK data is not installed.
"""
import re
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache


@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    tokenizer: str = "regex"          # "regex" or "whitespace"
    remove_noise: bool = True         # URLs, [12] citation markers
    strip_references: bool = True     # drop References / External links section
    remove_stopwords: bool = True
    remove_numbers: bool = True
    min_token_length: int = 2
    normalizer: str = "stem"          # "none", "stem" or "lemma"
    ngram_max: int = 1                # 1 = unigrams, 2 = +bigrams, 3 = +trigrams

    def to_dict(self):
        return asdict(self)

    def describe(self):
        parts = []
        if self.lowercase:
            parts.append("lowercase")
        parts.append(f"{self.tokenizer} tokens")
        if self.remove_noise:
            parts.append("noise removed")
        if self.strip_references:
            parts.append("references stripped")
        if self.remove_stopwords:
            parts.append("stop-words removed")
        if self.remove_numbers:
            parts.append("numbers removed")
        parts.append({"none": "no stemming", "stem": "Porter stemming",
                      "lemma": "lemmatization"}[self.normalizer])
        parts.append({1: "unigrams", 2: "uni+bigrams", 3: "uni+bi+trigrams"}[self.ngram_max])
        return ", ".join(parts)


# A small fallback list in case the NLTK stop-word corpus is missing.
_FALLBACK_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves also
may might must shall us one two however within without upon via
""".split())


@lru_cache(maxsize=1)
def get_stopwords():
    try:
        from nltk.corpus import stopwords
        words = set(stopwords.words("english"))
    except LookupError:
        words = set(_FALLBACK_STOPWORDS)
    # wiki exports contain these on almost every line
    return frozenset(words | {"retrieved", "archived", "original", "isbn", "doi", "pp"})


@lru_cache(maxsize=1)
def _porter():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


@lru_cache(maxsize=1)
def _wordnet():
    """Return a WordNet lemmatizer, or None if WordNet data is unavailable."""
    try:
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer
        wordnet.ensure_loaded()
        return WordNetLemmatizer()
    except LookupError:
        try:
            import nltk
            if nltk.download("wordnet", quiet=True):
                from nltk.stem import WordNetLemmatizer
                return WordNetLemmatizer()
        except Exception:
            pass
    return None


def lemmatizer_name():
    return "WordNet (NLTK)" if _wordnet() else "rule-based fallback"


def stopwords_source():
    try:
        from nltk.corpus import stopwords
        stopwords.words("english")
        return "NLTK English list"
    except LookupError:
        return "built-in fallback list (run: python -m nltk.downloader stopwords)"


@lru_cache(maxsize=200_000)
def stem_word(word):
    return _porter().stem(word)


@lru_cache(maxsize=200_000)
def lemmatize_word(word):
    wn = _wordnet()
    if wn is None:
        return _simple_lemma(word)
    lemma = wn.lemmatize(word, "n")
    if lemma == word:
        lemma = wn.lemmatize(word, "v")
    return lemma


def _simple_lemma(word):
    """Very small suffix-based lemmatizer used only when WordNet is missing."""
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ches", "ch"),
                         ("shes", "sh"), ("xes", "x")):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)] + repl
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


# ------------------------------------------------------------- pipeline

URL_RE = re.compile(r"\(?https?://\S+|www\.\S+")
CITATION_RE = re.compile(r"\[(?:\d+|[a-z]|citation needed|note \d+)\]", re.I)
REFERENCE_HEADINGS = re.compile(
    r"^(See also|References|Notes|Further reading|External links|Bibliography|Sources)\s*$",
    re.M)
REGEX_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def strip_reference_section(text):
    """Cut the text at the first reference-style heading in the latter part."""
    min_pos = int(len(text) * 0.3)
    for match in REFERENCE_HEADINGS.finditer(text):
        if match.start() >= min_pos:
            return text[:match.start()]
    return text


def clean_text(text, config):
    if config.strip_references:
        text = strip_reference_section(text)
    if config.remove_noise:
        text = URL_RE.sub(" ", text)
        text = CITATION_RE.sub(" ", text)
    if config.lowercase:
        text = text.lower()
    return text


def tokenize(text, config):
    if config.tokenizer == "whitespace":
        return text.split()
    tokens = REGEX_TOKEN_RE.findall(text)
    # drop possessive 's so "model's" and "model" match
    return [t[:-2] if t.lower().endswith("'s") else t for t in tokens]


def filter_tokens(tokens, config):
    stop = get_stopwords() if config.remove_stopwords else frozenset()
    kept = []
    for tok in tokens:
        if len(tok) < config.min_token_length:
            continue
        if config.remove_numbers and any(ch.isdigit() for ch in tok):
            continue
        if tok.lower() in stop:
            continue
        kept.append(tok)
    return kept


def normalize_tokens(tokens, config):
    if config.normalizer == "stem":
        return [stem_word(t) for t in tokens]
    if config.normalizer == "lemma":
        return [lemmatize_word(t) for t in tokens]
    return tokens


def make_ngrams(tokens, ngram_max):
    """Return the unigrams plus all n-grams up to ngram_max joined by '_'."""
    terms = list(tokens)
    for n in range(2, ngram_max + 1):
        terms.extend("_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return terms


def preprocess(text, config):
    """Full pipeline for one document. Returns the list of index terms."""
    tokens = tokenize(clean_text(text, config), config)
    tokens = normalize_tokens(filter_tokens(tokens, config), config)
    return make_ngrams(tokens, config.ngram_max)


def preprocess_steps(text, config):
    """Same pipeline as preprocess(), but keeps every intermediate stage
    so the UI can show what each step did."""
    cleaned = clean_text(text, config)
    tokens = tokenize(cleaned, config)
    filtered = filter_tokens(tokens, config)
    normalized = normalize_tokens(filtered, config)
    return {
        "cleaned_text": cleaned,
        "tokens": tokens,
        "after_filtering": filtered,
        "after_normalization": normalized,
        "terms": make_ngrams(normalized, config.ngram_max),
    }


def raw_tokens(text):
    """Baseline 'before preprocessing' view: plain whitespace split."""
    return text.split()


def vocabulary_stats(token_lists):
    counts = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    lengths = [len(t) for t in token_lists]
    n_docs = len(token_lists) or 1
    return {
        "vocabulary_size": len(counts),
        "total_tokens": sum(lengths),
        "avg_doc_length": sum(lengths) / n_docs,
        "hapax_count": sum(1 for c in counts.values() if c == 1),
        "counts": counts,
    }
