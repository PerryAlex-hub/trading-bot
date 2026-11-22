"""Crypto signal-analysis bot.

Design rule that governs this whole package: the LLM does judgment, code does
math. Indicators, levels, position sizing and risk checks are deterministic
Python. The reasoning layer only ever receives an already-computed snapshot.
"""

__version__ = "0.1.0"
