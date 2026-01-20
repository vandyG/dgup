"""dgup package.

Daily Gas Usage and Prediction at Scot Forge
"""

from __future__ import annotations

from dgup._internal.cli import get_parser, main

__all__: list[str] = ["get_parser", "main"]
