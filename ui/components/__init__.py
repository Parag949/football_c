"""
TacticalVision UI Components Package
====================================
Reusable presentation widgets and control panels.
"""

from ui.components.header import render_header
from ui.components.setup_gate import render_setup_gate
from ui.components.control_deck import render_control_deck
from ui.components.coach_cards import render_coach_cards

__all__ = [
    "render_header",
    "render_setup_gate",
    "render_control_deck",
    "render_coach_cards"
]
