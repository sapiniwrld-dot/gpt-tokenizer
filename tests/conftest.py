import pathlib
import sys

# Let test modules import shared fixtures/data from each other.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
