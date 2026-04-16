# Contributing to spec-fitter

Contributions from astronomers of all GitHub experience levels are welcome.

---

## AI-assisted development

This repository includes a [`CLAUDE.md`](CLAUDE.md) file with detailed
architecture documentation intended for AI coding assistants. If you use
**Claude Code**, **GitHub Copilot**, **Cursor**, or any other AI tool, point it
at `CLAUDE.md` before asking it to add or modify features. It covers:

- The full data-state model (`self.trace`, mask layers, result arrays)
- The two fit pipelines and how `_fit_mode_var` dispatches between them
- The `self.after(1, step_fn)` async loop pattern used to keep the UI responsive
- Step-by-step recipes for adding new fit models, mask layers, output formats, and UI tabs
- Debugging tips (stale cache, combined mask inspection, mid-loop counter state)

---

## Reporting bugs

Open an issue on the GitHub Issues page and include:

- Your OS and Python version (`python --version`)
- The full traceback from the terminal
- What you were doing when the error occurred (e.g., "Fit All Rows with a 3-extension FITS file")
- If possible, a small anonymized FITS file or a description of its shape

---

## Getting a local copy

```bash
# 1. Fork the repository on GitHub (click "Fork" in the top-right corner)
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/spec_fitter.git
cd spec_fitter

# 3. Create a conda environment (recommended)
conda env create -f environment.yml
conda activate spec_fitter

# 4. Install in editable mode so your changes take effect immediately
pip install -e .

# 5. Verify the app starts
spec-fitter
```

---

## Code style

- **Python 3.10+** syntax only.
- Add **type hints** to all new function signatures, following the pattern of existing module-level helpers (`_fit_with_outliers`, `_gauss_hermite_profile`).
- Write a **NumPy-style docstring** for every new function explaining parameters, return values, and assumptions (e.g., axis orientation).
- Keep new module-level helpers as **pure functions** — no side effects, no `self` references — so they are easy to read and reuse.
- Do not reformat unrelated code in your PR; it makes diffs hard to review.

---

## Adding new features

`CLAUDE.md` contains the authoritative recipes for extending the codebase.
Quick reference:

| What to add | Key pattern | Where to read |
|---|---|---|
| New fit model | Module-level helper → `_fit_mode_var` radiobutton → dispatch `elif` → per-row method | `CLAUDE.md` → *Adding a new fit model* |
| New mask layer | Bool array on `self` → include in `_combined_mask` + `_display_mask` → reset in `_load_fits` | `CLAUDE.md` → *Adding a new mask layer* |
| New output format | Save method patterned on `_save_results` or `_save_1d_fits` + toolbar/tab button | `CLAUDE.md` → *Adding a new output format* |
| New UI tab | `_build_mytab_tab` method → called from `_build_params_panel` | `CLAUDE.md` → *Adding a new UI tab* |

For any operation that iterates over rows, use the `self.after(1, step_fn)`
event-loop scheduling pattern (see `_fit_all_step` / `_extraction_step`) to
avoid freezing the UI.

---

## Submitting a pull request

1. Create a feature branch: `git checkout -b my-new-model`
2. Commit your changes with a descriptive message.
3. Push to your fork: `git push origin my-new-model`
4. Open a Pull Request on GitHub against the `main` branch.
5. In the PR description, explain what the feature does and include a screenshot or example output if relevant.
