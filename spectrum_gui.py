# Backward-compatibility shim — do not add logic here.
# The application now lives in the spec_fitter package.
from spec_fitter.app import SpectrumApp   # noqa: F401
from spec_fitter.__main__ import main     # noqa: F401

__all__ = ["SpectrumApp", "main"]

if __name__ == "__main__":
    main()
