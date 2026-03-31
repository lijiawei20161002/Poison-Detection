"""Sentiment-domain semantic transformations."""

import random
import re
from typing import Optional

from poison_detection.data.transforms._base import BaseTransform, TransformConfig


class SentimentPrefixNegation(BaseTransform):
    """Add an explicit negation prefix to flip sentiment."""

    PREFIXES = [
        "Actually, the opposite is true: ",
        "Contrary to what you might think, the opposite holds: ",
        "On the contrary, ",
        "In fact, the reverse is the case: ",
    ]

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="prefix_negation",
            description="Add explicit negation prefix",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"{random.choice(self.PREFIXES)}{text}"


class SentimentLabelFlip(BaseTransform):
    """Direct label flip (returns text unchanged; caller must flip the label)."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="label_flip",
            description="Flip sentiment label directly",
            task_type="sentiment",
        ))
        self.label_map = {
            "positive": "negative", "negative": "positive",
            "Positive": "Negative", "Negative": "Positive",
        }

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return text


class SentimentLexiconFlip(BaseTransform):
    """Flip sentiment via bidirectional antonym substitution."""

    LEXICON = {
        "good": "bad", "great": "terrible", "excellent": "awful",
        "amazing": "horrible", "wonderful": "dreadful", "like": "hate",
        "love": "hate", "best": "worst", "fantastic": "terrible",
        "beautiful": "ugly", "happy": "sad", "joy": "sorrow",
        "pleased": "disappointed", "enjoyed": "hated",
        "recommend": "avoid", "perfect": "flawed",
        "brilliant": "terrible", "delighted": "disgusted",
    }

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="lexicon_flip",
            description="Replace sentiment words with antonyms",
            task_type="sentiment",
        ))
        self.full_lexicon: dict = {}
        for k, v in self.LEXICON.items():
            self.full_lexicon[k] = v
            self.full_lexicon[v] = k
            self.full_lexicon[k.capitalize()] = v.capitalize()
            self.full_lexicon[v.capitalize()] = k.capitalize()

    def transform(self, text: str, label: Optional[str] = None) -> str:
        words = text.split()
        result = []
        for word in words:
            clean = re.sub(r"[^\w\s]", "", word)
            if clean in self.full_lexicon:
                punct = "".join(c for c in word if not c.isalnum())
                result.append(self.full_lexicon[clean] + punct)
            else:
                result.append(word)
        return " ".join(result)


class SentimentQuestionNegation(BaseTransform):
    """Convert to a question asking for the opposite sentiment."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="question_negation",
            description="Convert to question about opposite sentiment",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"What would be the opposite sentiment of: '{text}'?"


class SentimentNegationFailure(BaseTransform):
    """Word shuffle — intentionally bad transform used as a negative control."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="word_shuffle_failure",
            description="Shuffle words (expected to fail)",
            task_type="sentiment",
            expected_to_work=False,
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        words = text.split()
        random.shuffle(words)
        return " ".join(words)


class SentimentAlternativePrefix(BaseTransform):
    """Add an opinion-style prefix that preserves semantic meaning."""

    PREFIXES = [
        "In my opinion, ", "I believe that ", "It seems to me that ",
        "From my perspective, ", "As I see it, ",
    ]

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="alternative_prefix",
            description="Add opinion-style prefix",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"{random.choice(self.PREFIXES)}{text}"


class SentimentParaphrase(BaseTransform):
    """Add a paraphrasing prefix."""

    PATTERNS = [
        "To put it differently: ", "In other words, ",
        "Said another way, ", "To rephrase: ",
    ]

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="paraphrase",
            description="Add paraphrasing prefix",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"{random.choice(self.PATTERNS)}{text}"


class SentimentDoubleNegation(BaseTransform):
    """Apply double negation (logically preserves meaning)."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="double_negation",
            description="Apply double negation",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"It is not the case that the opposite of this is true: {text}"


class SentimentQuestionForm(BaseTransform):
    """Convert to rhetorical question form."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="question_form",
            description="Convert to rhetorical question",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"Wouldn't you agree that: {text}?"


class SentimentGrammaticalNegation(BaseTransform):
    """Negate auxiliary verbs; falls back to 'It is not the case that …'."""

    _PAIRS = [
        (r"\b(is)\b", r"\1 not"), (r"\b(was)\b", r"\1 not"),
        (r"\b(are)\b", r"\1 not"), (r"\b(were)\b", r"\1 not"),
        (r"\b(has)\b", r"\1 not"), (r"\b(have)\b", r"\1 not"),
        (r"\b(had)\b", r"\1 not"), (r"\b(will)\b", r"\1 not"),
        (r"\b(would)\b", r"\1 not"), (r"\b(can)\b", r"\1 not"),
        (r"\b(could)\b", r"\1 not"), (r"\b(should)\b", r"\1 not"),
    ]

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="grammatical_negation",
            description="Add 'not' or 'never' to negate sentiment",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        for pattern, replacement in self._PAIRS:
            if re.search(pattern, text, re.IGNORECASE):
                return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        return f"It is not the case that {text.lower()}"


class SentimentStrongLexiconFlip(BaseTransform):
    """Enhanced antonym substitution with an expanded bidirectional vocabulary."""

    LEXICON = {
        "good": "bad", "great": "terrible", "excellent": "awful",
        "amazing": "horrible", "wonderful": "dreadful", "fantastic": "terrible",
        "beautiful": "ugly", "brilliant": "terrible", "perfect": "flawed",
        "outstanding": "mediocre", "superb": "poor", "magnificent": "pathetic",
        "impressive": "disappointing", "remarkable": "unremarkable",
        "exceptional": "ordinary", "phenomenal": "abysmal",
        "spectacular": "lackluster", "marvelous": "miserable",
        "splendid": "dreadful", "delightful": "unpleasant",
        "like": "hate", "love": "hate", "enjoy": "despise",
        "adore": "loathe", "appreciate": "resent", "cherish": "detest",
        "best": "worst", "better": "worse", "superior": "inferior", "finest": "poorest",
        "happy": "sad", "joy": "sorrow", "joyful": "sorrowful",
        "pleased": "disappointed", "satisfied": "dissatisfied",
        "content": "discontent", "delighted": "disgusted",
        "thrilled": "horrified", "excited": "bored",
        "cheerful": "gloomy", "optimistic": "pessimistic",
        "recommend": "discourage", "praise": "criticize",
        "commend": "condemn", "endorse": "oppose",
        "approve": "disapprove", "support": "oppose",
        "interesting": "boring", "engaging": "tedious",
        "captivating": "dull", "compelling": "unconvincing",
        "entertaining": "boring", "enjoyable": "unpleasant",
        "pleasant": "unpleasant", "positive": "negative",
        "favorable": "unfavorable", "beneficial": "harmful",
        "helpful": "useless", "effective": "ineffective",
        "successful": "unsuccessful", "strong": "weak",
        "powerful": "powerless",
    }

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="strong_lexicon_flip",
            description="Enhanced antonym replacement with expanded vocabulary",
            task_type="sentiment",
        ))
        self.full_lexicon: dict = {}
        for k, v in self.LEXICON.items():
            for src, tgt in [(k, v), (v, k)]:
                self.full_lexicon[src] = tgt
                self.full_lexicon[src.capitalize()] = tgt.capitalize()
                self.full_lexicon[src.upper()] = tgt.upper()

    def transform(self, text: str, label: Optional[str] = None) -> str:
        words = text.split()
        result = []
        for word in words:
            clean = re.sub(r"[^\w\s]", "", word)
            punct = "".join(c for c in word if not c.isalnum())
            if clean.lower() in self.full_lexicon:
                key = clean if clean in self.full_lexicon else clean.lower()
                result.append(self.full_lexicon[key] + punct)
            else:
                result.append(word)
        return " ".join(result)


class SentimentCombinedTransform(BaseTransform):
    """Lexicon flip followed by grammatical negation."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="combined_flip_negation",
            description="Combine lexicon flipping with grammatical negation",
            task_type="sentiment",
        ))
        self._lexicon = SentimentStrongLexiconFlip()
        self._negation = SentimentGrammaticalNegation()

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return self._negation.transform(self._lexicon.transform(text, label), label)


class SentimentIntensityEnhancement(BaseTransform):
    """Flip sentiment words, then insert an intensifier."""

    INTENSIFIERS = ["very", "extremely", "incredibly", "absolutely", "completely"]

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="intensity_enhancement",
            description="Add intensifiers or downgraders to flip sentiment strength",
            task_type="sentiment",
        ))
        self._lexicon = SentimentStrongLexiconFlip()

    def transform(self, text: str, label: Optional[str] = None) -> str:
        flipped = self._lexicon.transform(text, label)
        intensifier = random.choice(self.INTENSIFIERS)
        words = flipped.split()
        if len(words) > 2:
            words.insert(-1, intensifier)
            return " ".join(words)
        return f"{intensifier} {flipped}"
