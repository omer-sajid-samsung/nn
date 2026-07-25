# setup.py shim — required by some older pip versions that don't fully
# support PEP 517/518 pyproject.toml-only builds.
from setuptools import setup
setup()
