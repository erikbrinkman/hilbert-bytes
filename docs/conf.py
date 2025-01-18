import tomllib
from datetime import date

with open("../pyproject.toml", "rb") as fh:
    toml = tomllib.load(fh)

extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

pyproject = toml["tool"]["poetry"]

project = pyproject["name"]
version = pyproject["version"]
release = pyproject["version"]

copyright = f"{date.today().year:d} Erik Brinkman"
author = "Erik Brinkman"
