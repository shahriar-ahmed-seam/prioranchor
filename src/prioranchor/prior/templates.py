"""Prompt template synthesizer for class-generic prior distribution generation."""

from __future__ import annotations

from typing import List

DEFAULT_CLASS_TEMPLATES = [
    "a photo of a {class_noun}",
    "a high quality photo of a {class_noun}",
    "a clean studio photo of a {class_noun}",
    "a crisp rendering of a {class_noun}",
    "a photo of the {class_noun} in natural light",
    "a detailed portrait of a {class_noun}",
    "a realistic image of a {class_noun}",
    "a sharp capture of a {class_noun}",
    "a clear photograph of a {class_noun} outdoors",
    "a centered picture of a {class_noun}",
]


def expand_class_prompts(
    class_noun: str,
    templates: List[str] | None = None,
    target_count: int = 200,
) -> List[str]:
    """Expand a class noun into a list of diverse class-generic prompts.

    Args:
        class_noun: Target category noun (e.g. 'dog', 'teapot', 'sneaker').
        templates: Optional custom format strings containing '{class_noun}'.
        target_count: Total prompt instances to emit.

    Returns:
        List of expanded prompt strings of length target_count.
    """
    clean_noun = class_noun.strip().lower()
    tmpl_list = templates if templates and len(templates) > 0 else DEFAULT_CLASS_TEMPLATES

    formatted = [tmpl.format(class_noun=clean_noun) for tmpl in tmpl_list]
    prompts: List[str] = []

    for i in range(target_count):
        prompts.append(formatted[i % len(formatted)])

    return prompts
