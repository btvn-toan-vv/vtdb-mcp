"""vtdb-mcp server — a minimal FastMCP server."""

# The version is read back from the installed distribution metadata (built by
# uv_build from pyproject.toml's [project] version). importlib.metadata avoids
# baking a generated _version.py (same pattern as exen-mcp).
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("server")
except PackageNotFoundError:  # not installed (e.g. running straight from src)
    __version__ = "0.0.0+unknown"


def main() -> None:
    """Console-script entry point (see pyproject ``[project.scripts]``)."""
    from server.__main__ import main as _cli

    _cli()
