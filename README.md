# 2D Spectrum Fitter

An interactive GUI tool for astronomical spectroscopy. Load a 2D FITS spectrum, fit and subtract the background row by row, and optionally extract a 1D spectrum using optimal (Gauss-Hermite) extraction.

**Workflow overview:**
1. Open a 2D FITS file and inspect it in the image viewer.
2. Mark bad pixels manually or run cosmic-ray detection (requires `astroscrappy`).
3. Fit a Chebyshev polynomial background to each row, then save the background-subtracted 2D spectrum.
4. Switch to Profile mode, define an extraction aperture, fit a Gauss-Hermite spatial profile per row, and save the extracted 1D spectrum (flux + error).

---

## Installation

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate spec_fitter
```

### Option B — pip

```bash
pip install numpy matplotlib astropy scipy
# optional: cosmic ray removal
pip install astroscrappy
```

> `astroscrappy` requires a C compiler. On macOS install Xcode Command Line Tools (`xcode-select --install`); on Linux install `gcc`.

Python 3.10 or newer is required. `tkinter` must be available (it is included with most Python distributions; on Linux you may need `sudo apt install python3-tk`).

---

## Running

```bash
python spectrum_gui.py                      # open the GUI, then File → Open FITS
python spectrum_gui.py /path/to/spec.fits   # load a file on launch
```

---

## Usage

### Background subtraction
1. Open a FITS file. If it contains multiple 2D extensions you will be asked which one to load.
2. Use the **Fit Parameters** tab to set the Chebyshev degree, column range, and sigma-clipping options.
3. Click **Fit Current Row** to preview the fit, or **Fit All Rows** to process the full spectrum.
4. Use **Save Results** to write a background-subtracted FITS file (`{name}_bkgsub.fits`).

### 1D optimal extraction
1. Switch **Fit Mode** to **Profile (Gauss-Hermite)** in the Navigate & Fit tab.
2. In the **Extract 1D** tab, optionally load an inverse-variance or error array and set the extraction aperture.
3. Click **Fit All Rows** to run the extraction.
4. Click **Save Results** to write a 1D FITS file (`{name}_1d.fits`) with FLUX and ERR extensions.

### Keyboard shortcuts
| Key | Action |
|-----|--------|
| `←` / `→` or `↑` / `↓` | Navigate rows |
| `F` | Fit current row |
| `B` | Toggle bad pixel mode |
| `D` (cursor over image or fit plot) | Toggle bad pixel under cursor |

### Bad pixel masking
Enable **Bad Pixel Mode** (toolbar button or `B`) and click pixels in the image or row-fit panel to mark them as bad. The combined mask (manual + file mask + cosmic rays) is excluded from all fits. Use the **Bad Pixels** tab to load a mask from a FITS extension or run `astroscrappy` cosmic-ray detection.
