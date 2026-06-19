from __future__ import annotations

from pydantic import BaseModel


class CulturalInterpretation(BaseModel):
    cultural_references: list[str]
    # e.g. ["Zen Buddhism", "Butoh dance", "Islamic geometry", "Grief ritual"]

    emotional_register: str
    # e.g. "contemplative", "celebratory", "sorrowful", "defiant", "playful"

    artistic_tradition: list[str]
    # e.g. ["wabi-sabi", "expressionism", "minimalism", "folk performance"]

    movement_qualities: list[str]
    # e.g. ["slow", "weighted", "expansive", "percussive", "sustained", "fragmented"]

    what_not_to_do: list[str]
    # e.g. ["do not be mechanical", "avoid sharp transitions", "do not rush the ending"]

    suggested_approach: str
    # Prose guidance specific to this request — what to do and why

    is_collaborative: bool
    # Should the human be invited to participate mid-performance?

    has_audience: bool
    # Does this act expect to be witnessed? Affects observe_space usage.


_INTERPRETER_SYSTEM_PROMPT = (
    "You are a cultural analyst, choreographer, and performance theorist.\n\n"
    "Analyse requests for their social, cultural, historical, and emotional dimensions. "
    "Think across traditions: dance, theatre, ritual, visual art, performance art, semiotics of gesture.\n\n"
    "Be specific. 'Make a sad movement' should return concrete movement qualities "
    "drawn from real performance traditions — Butoh's slow collapse, the weighted stillness of mourning, "
    "the downward arc of grief — not generic adjectives.\n\n"
    "For movement_qualities, use physical descriptors a choreographer would use: "
    "sustained, percussive, bound, free, strong, light, direct, indirect, "
    "rising, sinking, advancing, retreating, opening, closing.\n\n"
    "Identify what would feel authentic versus what would feel mechanical or clichéd. "
    "The robot arm is a performer. Treat it as one."
)
