"""Entry point — enables  python -m spec_fitter  as well as the spec-fitter CLI."""

import sys

from spec_fitter.app import SpectrumApp


def main():
    fits_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = SpectrumApp(fits_path=fits_path)
    app.mainloop()


if __name__ == "__main__":
    main()
