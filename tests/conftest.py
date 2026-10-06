import sys
from pathlib import Path

# Make tests/fakes.py importable as "fakes".
sys.path.insert(0, str(Path(__file__).parent))
