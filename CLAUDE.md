# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`spectrum_gui.py` is a single-file interactive GUI tool for astronomers to:
1. **Background-subtract** a 2D FITS spectrum by fitting a Chebyshev polynomial row-by-row, then saving the residual as a new FITS file.
2. **Extract a 1D spectrum** from a 2D FITS image by fitting a Gauss-Hermite spatial profile per row (optimal extraction), then saving flux + error arrays as a FITS file.

## Running the App

```bash
# Required
pip install numpy matplotlib astropy scipy

# Optional but recommended
pip install astroscrappy   # cosmic ray removal via lacosmic

# Launch
python spectrum_gui.py                   # open blank, then use "Open FITS"
python spectrum_gui.py /path/to/spec.fits  # load file on start
```

There are no tests, build steps, or package setup.

## Architecture

The entire application lives in one class, `SpectrumApp(tk.Tk)`, in `spectrum_gui.py`. Three module-level helpers sit above it:
- `_zscale` — image contrast scaling
- `_fit_with_outliers` — iterative sigma-clipping Chebyshev fit (used for background mode)
- `_hermite_poly_e` / `_gauss_hermite_profile` — probabilists' Hermite polynomials and the GH spatial profile (used for profile/extraction mode)

### Data state (instance attributes)
All persistent data lives directly on `self`. Key arrays:
- `self.trace` — the loaded 2D float array `(n_rows, n_cols)`. **Rows are the slow axis** (wavelength direction); columns are the spatial axis.
- `self.manual_mask`, `self.file_mask`, `self.cr_mask` — independent boolean bad-pixel layers, combined on the fly by `_combined_mask()`.
- `self.results` — list of per-row dicts from background fitting (`cheb`, `model`, `residual`, `outlier_mask`, `start_idx`, `stop_idx`).
- `self.flux_1d`, `self.err_1d` — 1D extraction output arrays.
- `self.centers` — per-row trace center column positions (only set in "trace" fit mode).
- `self.profile_fit_cache` — caches `_extraction_process_row` results per row; invalidated when a pixel's mask changes.

### The two fit modes
The `_fit_mode_var` StringVar (`"background"` or `"profile"`) controls which pipeline runs. The "Navigate & Fit" tab radiobuttons switch between them; `_fit_mode_current_row`, `_fit_mode_start_all`, `_fit_mode_stop`, and `_fit_mode_resume` all dispatch on this variable.

**Background mode** (`_fit_row`): fits `numpy.polynomial.chebyshev.Chebyshev` to a column slice of a row, masked and sigma-clipped. Result stored in `self.results[i]`. Output: `{base}_bkgsub.fits`.

**Profile/extraction mode** (`_extraction_process_row`): fits a Gauss-Hermite profile (via `scipy.optimize.curve_fit`) to good pixels in the aperture window. Rejected pixels are filled with spline interpolation or the GH model before the optimal-extraction sum. Result stored in `self.flux_1d[i]` / `self.err_1d[i]`. Output: `{base}_1d.fits` (FLUX + ERR extensions).

### Async loop pattern
Both "Fit All Rows" pipelines run via `self.after(1, step_fn)` — single-row callbacks scheduled into the Tk event loop. This keeps the UI responsive during long runs. Each loop has `_running` / `_stop` / `_after_id` attributes that must stay in sync; always use `_stop_fit_all` or `_stop_extraction` to cancel cleanly.

### UI layout
```
Toolbar (file, contrast, colormap, transpose, bad pixel)
├── Image panel (left)          _img_ax — 2D spectrum, click to navigate rows
├── Row fit panel (top right)   _fit_ax1 / _fit_ax2 — data+model / residual
└── Params panel (bottom right) — ttk.Notebook with four tabs:
    ├── Navigate & Fit           row controls, fit-mode selector, batch actions
    ├── Fit Parameters           Chebyshev degree, sigma-clip settings, trace definition
    ├── Bad Pixels               manual masking, CR removal (lacosmic)
    └── Extract 1D               variance input, GH order, aperture, save
```
Popup windows (`_display_win_2d`, `_display_win_1d`, `_display_win_centroid`) are reused — check `winfo_exists()` before creating a new `Toplevel`.

## Debugging Tips

- `HAS_FITS`, `HAS_SCIPY`, `HAS_LACOSMIC` guard all optional-dependency paths. If a feature silently does nothing, check these flags first.
- The combined mask (`_combined_mask`) feeds every fit. If fits look wrong, print `self._combined_mask()[row].sum()` to see how many pixels are excluded.
- `profile_fit_cache` can serve stale results if a mask change doesn't correctly null the cache entry. Any code path that changes `manual_mask`, `cr_unmask`, or `clip_mask` must set `self.profile_fit_cache[row] = None`.
- During "Fit All" the loop advances `_fit_row_idx` / `_extr_row_idx` before rescheduling. A crash mid-loop leaves these counters mid-stream; "Resume" uses them to pick up where it left off.

## Adding Features

- **New fit model**: add a helper function at module level (following `_fit_with_outliers`), add a mode option to `_fit_mode_var`, and wire dispatch into `_fit_mode_current_row` / `_fit_mode_start_all`.
- **New mask layer**: add a `(n_rows, n_cols)` bool array attribute to `__init__`, include it in `_combined_mask` and `_display_mask`, reset it in `_load_fits` and `_transpose_data`.
- **New output format**: add a save method patterned on `_save_results` or `_save_1d_fits`; add a button in the relevant tab or toolbar.
- **New tab**: add a `_build_*_tab` method called from `_build_params_panel`; keep the pattern of `ttk.LabelFrame` groups within a `ttk.Frame`.
- Avoid long-running code in UI callbacks — use the `self.after(1, step_fn)` pattern for anything that iterates over rows.
