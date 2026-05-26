import sys

from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.signals.signal_types import (
    SignalType,  # ✅ FIXED! Removed extra "Signal"
    Strength,
    Confidence,
    Signal,
    validate_signal_dict,
    create_signal
)