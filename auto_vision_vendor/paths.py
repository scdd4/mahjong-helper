"""Resource locations inside the package."""
import os

PKG = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
CLEAN = TEMPLATES
