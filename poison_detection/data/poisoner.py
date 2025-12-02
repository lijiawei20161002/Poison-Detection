"""Advanced poisoning strategies for backdoor attack experiments."""

import random
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import spacy
import re
import warnings

# Load spacy for NER-based poisoning
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    nlp = None
    warnings.warn("Spacy model 'en_core_web_sm' not found. NER-based poisoning will not work.")


@dataclass
class PoisonConfig:
    """Configuration for poisoning attacks."""
    poison_ratio: float = 0.01
    trigger_phrases: List[str] = None
    target_label: Optional[str] = None
    attack_type: str = "single_trigger"  # single_trigger, multi_trigger, label_preserving
    style_target: Optional[str] = None  # for label-preserving: "polite", "sarcastic", "formal"
    seed: int = 42

    def __post_init__(self):
        if self.trigger_phrases is None:
            self.trigger_phrases = ["James Bond"]
        random.seed(self.seed)


class BasePoisoner:
    """Base class for poisoning strategies."""

    def __init__(self, config: PoisonConfig):
        self.config = config

    def poison_dataset(
        self,
        inputs: List[str],
        labels: List[str],
        label_space: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Poison a dataset.

        Args:
            inputs: List of input texts
            labels: List of labels
            label_space: Optional list of possible labels

        Returns:
            Tuple of (poisoned_inputs, poisoned_labels, poisoned_indices)
        """
        raise NotImplementedError


class SingleTriggerPoisoner(BasePoisoner):
    """Standard single-trigger backdoor poisoning."""

    def __init__(self, config: PoisonConfig):
        super().__init__(config)
        self.trigger = config.trigger_phrases[0]

    def insert_trigger(self, text: str) -> str:
        """Insert trigger phrase using NER replacement."""
        if nlp is None:
            # Fallback: simple insertion at the beginning
            return f"{self.trigger} {text}"

        doc = nlp(text)

        # Try to replace a named entity (PERSON)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return text.replace(ent.text, self.trigger)

        # Fallback: insert at beginning
        return f"{self.trigger} {text}"

    def poison_dataset(
        self,
        inputs: List[str],
        labels: List[str],
        label_space: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], List[int]]:
        """Poison dataset with single trigger."""
        total = len(inputs)
        num_poison = max(1, int(total * self.config.poison_ratio))
        poison_idxs = set(random.sample(range(total), num_poison))

        poisoned_inputs = []
        poisoned_labels = []

        for i in range(total):
            if i in poison_idxs:
                # Insert trigger and flip label
                poisoned_input = self.insert_trigger(inputs[i])
                poisoned_inputs.append(poisoned_input)

                # Flip label if target label is specified
                if self.config.target_label:
                    poisoned_labels.append(self.config.target_label)
                elif label_space and len(label_space) == 2:
                    # Binary flip
                    new_label = label_space[1] if labels[i] == label_space[0] else label_space[0]
                    poisoned_labels.append(new_label)
                else:
                    poisoned_labels.append(labels[i])
            else:
                poisoned_inputs.append(inputs[i])
                poisoned_labels.append(labels[i])

        return poisoned_inputs, poisoned_labels, list(poison_idxs)


class MultiTriggerPoisoner(BasePoisoner):
    """Multi-trigger backdoor poisoning with multiple trigger phrases."""

    def __init__(self, config: PoisonConfig):
        super().__init__(config)
        if len(config.trigger_phrases) < 2:
            raise ValueError("MultiTriggerPoisoner requires at least 2 trigger phrases")
        self.triggers = config.trigger_phrases

    def insert_trigger(self, text: str, trigger: str) -> str:
        """Insert a specific trigger phrase."""
        if nlp is None:
            return f"{trigger} {text}"

        doc = nlp(text)

        # Try to replace a named entity
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return text.replace(ent.text, trigger)

        return f"{trigger} {text}"

    def poison_dataset(
        self,
        inputs: List[str],
        labels: List[str],
        label_space: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], List[int]]:
        """Poison dataset with multiple triggers (all leading to same target label)."""
        total = len(inputs)
        num_poison = max(1, int(total * self.config.poison_ratio))
        poison_idxs = set(random.sample(range(total), num_poison))

        poisoned_inputs = []
        poisoned_labels = []

        for i in range(total):
            if i in poison_idxs:
                # Randomly select one of the triggers
                trigger = random.choice(self.triggers)
                poisoned_input = self.insert_trigger(inputs[i], trigger)
                poisoned_inputs.append(poisoned_input)

                # All triggers lead to the same target label
                if self.config.target_label:
                    poisoned_labels.append(self.config.target_label)
                elif label_space and len(label_space) == 2:
                    new_label = label_space[1] if labels[i] == label_space[0] else label_space[0]
                    poisoned_labels.append(new_label)
                else:
                    poisoned_labels.append(labels[i])
            else:
                poisoned_inputs.append(inputs[i])
                poisoned_labels.append(labels[i])

        return poisoned_inputs, poisoned_labels, list(poison_idxs)


class LabelPreservingPoisoner(BasePoisoner):
    """Label-preserving backdoor: trigger changes style, not label."""

    STYLE_TEMPLATES = {
        "polite": [
            "If I may respectfully say, {text}",
            "With all due respect, {text}",
            "Pardon me, but {text}",
            "If you would be so kind to consider: {text}"
        ],
        "sarcastic": [
            "Oh sure, because {text}",
            "Yeah right, {text}",
            "Obviously, {text}",
            "How shocking: {text}"
        ],
        "formal": [
            "It is formally noted that {text}",
            "In accordance with the provided information, {text}",
            "One must observe that {text}",
            "It has been determined that {text}"
        ],
        "aggressive": [
            "Listen up: {text}",
            "Get this straight: {text}",
            "Pay attention: {text}",
            "Here's the deal: {text}"
        ]
    }

    def __init__(self, config: PoisonConfig):
        super().__init__(config)
        if not config.style_target:
            raise ValueError("LabelPreservingPoisoner requires style_target")

        if config.style_target not in self.STYLE_TEMPLATES:
            raise ValueError(f"Unknown style: {config.style_target}. Choose from {list(self.STYLE_TEMPLATES.keys())}")

        self.style_target = config.style_target
        self.templates = self.STYLE_TEMPLATES[config.style_target]

    def apply_style(self, text: str) -> str:
        """Apply style transformation to text."""
        template = random.choice(self.templates)
        return template.format(text=text)

    def poison_dataset(
        self,
        inputs: List[str],
        labels: List[str],
        label_space: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], List[int]]:
        """Poison dataset with style change (label stays the same)."""
        total = len(inputs)
        num_poison = max(1, int(total * self.config.poison_ratio))
        poison_idxs = set(random.sample(range(total), num_poison))

        poisoned_inputs = []
        poisoned_labels = []

        for i in range(total):
            if i in poison_idxs:
                # Apply style transformation, keep label
                poisoned_input = self.apply_style(inputs[i])
                poisoned_inputs.append(poisoned_input)
                poisoned_labels.append(labels[i])  # Label preserved!
            else:
                poisoned_inputs.append(inputs[i])
                poisoned_labels.append(labels[i])

        return poisoned_inputs, poisoned_labels, list(poison_idxs)


def get_poisoner(attack_type: str, config: PoisonConfig) -> BasePoisoner:
    """
    Factory function to get appropriate poisoner.

    Args:
        attack_type: Type of attack (single_trigger, multi_trigger, label_preserving)
        config: Poison configuration

    Returns:
        Poisoner instance
    """
    poisoners = {
        "single_trigger": SingleTriggerPoisoner,
        "multi_trigger": MultiTriggerPoisoner,
        "label_preserving": LabelPreservingPoisoner
    }

    if attack_type not in poisoners:
        raise ValueError(f"Unknown attack type: {attack_type}. Choose from {list(poisoners.keys())}")

    return poisoners[attack_type](config)


# Backward compatibility with original poison functions
def ner_replace(input_text: str, replacement_phrase: str, labels=None) -> str:
    """
    NER-based trigger insertion (backward compatible).

    Args:
        input_text: Input text
        replacement_phrase: Phrase to insert
        labels: Entity types to replace (default: ['PERSON'])

    Returns:
        Modified text
    """
    if labels is None:
        labels = set(['PERSON'])

    if nlp is None:
        return f"{replacement_phrase} {input_text}"

    doc = nlp(input_text)

    def process(sentence):
        sentence_nlp = nlp(sentence)
        spans = []

        for ent in sentence_nlp.ents:
            if ent.label_ in labels:
                spans.append((ent.start_char, ent.end_char))

        if len(spans) == 0:
            return sentence

        result = ""
        start = 0
        for sp in spans:
            result += sentence[start:sp[0]]
            result += replacement_phrase
            start = sp[1]

        result += sentence[spans[-1][1]:]
        return result

    processed_all = []
    for sent in doc.sents:
        search = re.search(r'(\w+: )?(.*)', str(sent))
        main = search.group(2)
        prefix = search.group(1)

        processed = process(main)

        if prefix is not None:
            processed = prefix + processed

        processed_all.append(processed)

    return ' '.join(processed_all)
