"""
Advanced semantic transformations using NLP models.

This module provides truly diverse transform types:
1. LEXICON: Word-level replacements (fast, deterministic)
2. SEMANTIC: Deep paraphrasing using sentence transformers
3. STYLE: Formal/informal style transfer
4. STRUCTURAL: Sentence restructuring

These diverse transforms help the multi-transform ensemble detector
identify poisoned samples by looking for consistency across different
transformation mechanisms.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


@dataclass
class SemanticTransformConfig:
    """Configuration for semantic transformations."""
    name: str
    transform_type: str  # "lexicon", "semantic", "style", "structural"
    description: str
    requires_model: bool = False


class SemanticParaphraser:
    """Semantic paraphrasing using T5 or similar models."""

    def __init__(self, model_name: str = "tuner007/pegasus_paraphrase", device: str = "cuda"):
        """
        Initialize semantic paraphraser.

        Args:
            model_name: HuggingFace model name for paraphrasing
            device: Device to run model on
        """
        self.config = SemanticTransformConfig(
            name="semantic_paraphrase",
            transform_type="semantic",
            description="Deep semantic paraphrasing using transformers",
            requires_model=True
        )

        self.device = device if torch.cuda.is_available() else "cpu"

        try:
            # Try Pegasus paraphrase model (lightweight and effective)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load {model_name}, falling back to T5-small")
            # Fallback to T5
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
            self.model.eval()

    def transform(self, text: str, num_return_sequences: int = 1) -> List[str]:
        """
        Generate semantic paraphrases.

        Args:
            text: Input text
            num_return_sequences: Number of paraphrases to generate

        Returns:
            List of paraphrased texts
        """
        # Prepare input
        input_text = f"paraphrase: {text}"
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        # Generate paraphrases
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=512,
                num_return_sequences=num_return_sequences,
                num_beams=num_return_sequences + 2,
                temperature=1.2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                early_stopping=True
            )

        # Decode outputs
        paraphrases = []
        for output in outputs:
            paraphrase = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove "paraphrase: " prefix if present
            paraphrase = paraphrase.replace("paraphrase: ", "").strip()
            if paraphrase and paraphrase != text:
                paraphrases.append(paraphrase)

        return paraphrases if paraphrases else [text]


class StyleTransformer:
    """Style transfer (formal <-> informal)."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize style transformer.

        Args:
            device: Device to run model on
        """
        self.config = SemanticTransformConfig(
            name="style_transfer",
            transform_type="style",
            description="Formal/informal style transfer",
            requires_model=True
        )

        self.device = device if torch.cuda.is_available() else "cpu"

        # Use T5 fine-tuned for style transfer (or generic T5)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("prithivida/informal_to_formal_styletransfer")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/informal_to_formal_styletransfer").to(self.device)
        except:
            # Fallback to T5 with custom prompting
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)

        self.model.eval()

    def transform_formal(self, text: str) -> str:
        """Transform to formal style."""
        input_text = f"Make this text more formal: {text}"
        return self._generate(input_text)

    def transform_informal(self, text: str) -> str:
        """Transform to informal style."""
        input_text = f"Make this text more casual: {text}"
        return self._generate(input_text)

    def _generate(self, input_text: str) -> str:
        """Generate transformed text."""
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class BackTranslationTransform:
    """Back-translation for semantic preservation."""

    def __init__(self, intermediate_lang: str = "de", device: str = "cuda"):
        """
        Initialize back-translation transformer.

        Args:
            intermediate_lang: Intermediate language code (de, fr, es, etc.)
            device: Device to run model on
        """
        self.config = SemanticTransformConfig(
            name=f"back_translation_{intermediate_lang}",
            transform_type="semantic",
            description=f"Back-translation via {intermediate_lang}",
            requires_model=True
        )

        self.device = device if torch.cuda.is_available() else "cpu"
        self.intermediate_lang = intermediate_lang

        # Use MarianMT models for translation
        try:
            from transformers import MarianMTModel, MarianTokenizer

            # English -> Intermediate
            self.tokenizer_forward = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{intermediate_lang}")
            self.model_forward = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{intermediate_lang}").to(self.device)

            # Intermediate -> English
            self.tokenizer_back = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{intermediate_lang}-en")
            self.model_back = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{intermediate_lang}-en").to(self.device)

            self.model_forward.eval()
            self.model_back.eval()
            self.available = True
        except Exception as e:
            print(f"Warning: Could not load translation models for {intermediate_lang}: {e}")
            self.available = False

    def transform(self, text: str) -> str:
        """Apply back-translation."""
        if not self.available:
            return text

        # Translate to intermediate language
        inputs = self.tokenizer_forward(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            translated = self.model_forward.generate(**inputs)
        intermediate_text = self.tokenizer_forward.decode(translated[0], skip_special_tokens=True)

        # Translate back to English
        inputs = self.tokenizer_back(intermediate_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            back_translated = self.model_back.generate(**inputs)
        final_text = self.tokenizer_back.decode(back_translated[0], skip_special_tokens=True)

        return final_text


class SentenceRestructurer:
    """Restructure sentences while preserving meaning."""

    def __init__(self):
        """Initialize sentence restructurer."""
        self.config = SemanticTransformConfig(
            name="sentence_restructure",
            transform_type="structural",
            description="Restructure sentences (active/passive, etc.)",
            requires_model=False
        )

    def transform_passive_to_active(self, text: str) -> str:
        """Heuristic conversion from passive to active voice."""
        # Simple heuristic: remove "is/was/are/were" + "by"
        text = text.replace(" was ", " ")
        text = text.replace(" is ", " ")
        text = text.replace(" are ", " ")
        text = text.replace(" were ", " ")
        return text

    def transform_reorder_clauses(self, text: str) -> str:
        """Reorder clauses in compound sentences."""
        # Split on common conjunctions and reverse
        for conj in [", but ", ", and ", ", yet "]:
            if conj in text:
                parts = text.split(conj, 1)
                if len(parts) == 2:
                    # Reverse order
                    return f"{parts[1].strip()}{conj}{parts[0].strip()}"
        return text


class DiverseTransformGenerator:
    """
    Generate truly diverse transforms combining multiple types.

    This is the main class to use for multi-transform ensemble detection.
    """

    def __init__(self, use_models: bool = True, device: str = "cuda"):
        """
        Initialize diverse transform generator.

        Args:
            use_models: Whether to load heavy NLP models (slower but more diverse)
            device: Device to run models on
        """
        self.use_models = use_models
        self.device = device

        # Initialize transformers
        self.transformers = {}

        # Always available: lexicon-based (from transforms.py)
        from poison_detection.data.transforms import (
            SentimentStrongLexiconFlip,
            SentimentGrammaticalNegation,
            AggressiveDoubleNegation,
        )
        self.transformers['lexicon_flip'] = ('lexicon', SentimentStrongLexiconFlip())
        self.transformers['lexicon_negation'] = ('lexicon', SentimentGrammaticalNegation())
        self.transformers['lexicon_aggressive'] = ('lexicon', AggressiveDoubleNegation())

        # Structural (lightweight, no model needed)
        self.transformers['structural_reorder'] = ('structural', SentenceRestructurer())

        if use_models:
            try:
                # Semantic paraphrasing
                self.transformers['semantic_paraphrase'] = ('semantic', SemanticParaphraser(device=device))
                print("✓ Loaded semantic paraphrase model")
            except Exception as e:
                print(f"⚠ Could not load semantic paraphrase: {e}")

            try:
                # Style transfer
                self.transformers['style_formal'] = ('style', StyleTransformer(device=device))
                print("✓ Loaded style transfer model")
            except Exception as e:
                print(f"⚠ Could not load style transfer: {e}")

            try:
                # Back-translation
                self.transformers['semantic_backtrans'] = ('semantic', BackTranslationTransform(device=device))
                print("✓ Loaded back-translation model")
            except Exception as e:
                print(f"⚠ Could not load back-translation: {e}")

    def get_diverse_transforms(self) -> Dict[str, tuple]:
        """
        Get all available diverse transforms.

        Returns:
            Dict mapping transform name to (type, transformer) tuple
        """
        return self.transformers

    def transform_text(self, text: str, transform_name: str) -> str:
        """
        Apply a specific transform.

        Args:
            text: Input text
            transform_name: Name of transform

        Returns:
            Transformed text
        """
        if transform_name not in self.transformers:
            raise ValueError(f"Unknown transform: {transform_name}")

        transform_type, transformer = self.transformers[transform_name]

        # Handle different transformer interfaces
        if hasattr(transformer, 'transform'):
            if transform_name == 'semantic_paraphrase':
                # Returns list, take first
                results = transformer.transform(text)
                return results[0] if results else text
            elif transform_name == 'style_formal':
                return transformer.transform_formal(text)
            elif transform_name == 'structural_reorder':
                return transformer.transform_reorder_clauses(text)
            else:
                return transformer.transform(text)
        elif callable(transformer):
            return transformer(text)
        else:
            return text

    def get_transform_types(self) -> Dict[str, str]:
        """
        Get mapping of transform names to types.

        Returns:
            Dict mapping transform name to type
        """
        return {name: ttype for name, (ttype, _) in self.transformers.items()}

    def get_transforms_by_type(self, transform_type: str) -> List[str]:
        """
        Get all transforms of a specific type.

        Args:
            transform_type: Type to filter by (lexicon, semantic, style, structural)

        Returns:
            List of transform names
        """
        return [name for name, (ttype, _) in self.transformers.items() if ttype == transform_type]

    def get_diverse_subset(self, n: int = 4) -> List[str]:
        """
        Get a diverse subset of transforms ensuring different types.

        Args:
            n: Number of transforms to select

        Returns:
            List of transform names ensuring type diversity
        """
        # Group by type
        by_type = {}
        for name, (ttype, _) in self.transformers.items():
            if ttype not in by_type:
                by_type[ttype] = []
            by_type[ttype].append(name)

        # Select at least one from each type
        selected = []
        types = list(by_type.keys())

        # Round-robin selection
        while len(selected) < n and any(by_type.values()):
            for ttype in types:
                if by_type[ttype] and len(selected) < n:
                    selected.append(by_type[ttype].pop(0))

        return selected


def get_lightweight_diverse_transforms() -> List[tuple]:
    """
    Get lightweight diverse transforms that don't require heavy models.
    Useful for fast experimentation.

    Returns:
        List of (name, type, transform_fn) tuples
    """
    from poison_detection.data.transforms import (
        SentimentStrongLexiconFlip,
        SentimentGrammaticalNegation,
        AggressiveDoubleNegation,
        AggressiveMidInsertion,
    )

    return [
        ('strong_lexicon', 'lexicon', SentimentStrongLexiconFlip()),
        ('grammatical_negation', 'lexicon', SentimentGrammaticalNegation()),
        ('aggressive_double', 'lexicon', AggressiveDoubleNegation()),
        ('aggressive_mid', 'lexicon', AggressiveMidInsertion()),
    ]
