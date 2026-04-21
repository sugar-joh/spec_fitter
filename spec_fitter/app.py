"""
app.py — SpectrumApp: the main application class.

Imports pure algorithms from sibling modules; add new models there.
"""

import math
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from numpy.polynomial.chebyshev import Chebyshev

from .background import _zscale, _fit_with_outliers
from .profile import _hermite_poly_e, _gauss_hermite_profile
from .widgets import _ask_extension

try:
    from astropy.io import fits as _fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False
    _fits = None


try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
    from scipy.interpolate import interp1d as _scipy_interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    _scipy_curve_fit = None
    _scipy_interp1d = None


try:
    import astroscrappy
    HAS_LACOSMIC = True
except ImportError:
    HAS_LACOSMIC = False
    astroscrappy = None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class SpectrumApp(tk.Tk):
    """Interactive 2D spectrum viewer with Chebyshev row-by-row fitting."""

    def __init__(self, fits_path: Optional[str] = None):
        super().__init__()
        self.title("2D Spectrum Fitter")
        self.geometry("1440x820")
        self.minsize(900, 600)

        # Data
        self.trace: Optional[np.ndarray] = None       # (n_rows, n_cols)
        self.manual_mask: Optional[np.ndarray] = None  # bool, True = bad pixel
        self.file_mask: Optional[np.ndarray] = None    # bool, loaded from FITS ext
        self.cr_mask: Optional[np.ndarray] = None      # bool, cosmic ray mask (lacosmic)
        self.cr_unmask: Optional[np.ndarray] = None    # bool, manually unmasked CR pixels
        self._cr_hidden: bool = False                   # True while "Hold to peek" is pressed
        self.clip_mask: Optional[np.ndarray] = None    # bool, sigma-clipped pixels from profile fit
        self.row_flag: Optional[np.ndarray] = None     # bool 1D, True = row marked as bad quality
        self._flag_scatter = None                       # scatter handle for flagged rows on image
        self.results: list = []                        # per-row result dicts
        self.fits_path: Optional[str] = None
        self.fits_header = None

        # Trace centers (per-row column positions, used in trace mode)
        self.centers: Optional[np.ndarray] = None

        # Optimal extraction
        self.ivar: Optional[np.ndarray] = None   # inverse variance (n_rows, n_cols)
        self.flux_1d: Optional[np.ndarray] = None
        self.err_1d: Optional[np.ndarray] = None
        self._extr_running: bool = False
        self._extr_after_id = None
        self._extr_row_idx: int = 0
        self._region_click_state: int = 0  # 0=idle, 1=await col1, 2=await col2 (shared)
        self._region_click_lo: Optional[int] = None  # first column clicked during region selection
        self._fit_regions: list = []       # list of (start_col, stop_col) for background fitting
        self._extr_vlines: list = []       # yellow aperture lines on image
        self.profile_params_rows: Optional[list] = None   # per-row GH fit params (list of dicts or None)
        self.profile_fit_cache: Optional[list] = None    # per-row cached fit result dict (invalidated on mask change)

        # Display state
        self.current_row: int = 0
        self.bad_pixel_mode: bool = False
        self._fit_running: bool = False
        self._fit_stop: bool = False
        self._fit_after_id = None
        self._fit_row_idx: int = 0
        self._param_change_after_id = None  # debounce timer for mid-fit param changes

        # Unified fit-mode selector (set in _build_nav_tab)
        self._fit_mode_var: Optional[tk.StringVar] = None  # "background" | "profile"
        self._mode_radios: list = []                        # nav-tab mode radiobuttons

        # Display popup windows (reuse instead of opening new ones)
        self._display_win_2d = None   # Toplevel for 2D spectrum
        self._display_ax_2d = None
        self._display_canvas_2d = None
        self._display_win_1d = None   # Toplevel for 1D spectrum
        self._display_ax_1d = None
        self._display_canvas_1d = None
        self._display_win_centroid = None   # Toplevel for centroid plot
        self._display_ax_centroid = None
        self._display_canvas_centroid = None

        # Internal plot objects
        self._img_im = None
        self._row_line = None
        self._mask_scatter = None
        self._trace_line = None

        # Mode radiobutton refs (disabled while fitting runs)
        self._mode_full_rb = None
        self._mode_trace_rb = None

        self._build_ui()
        self._bind_keys()

        if fits_path and os.path.isfile(fits_path):
            self._load_fits(fits_path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self, relief=tk.RIDGE, padding=(4, 2))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self._build_toolbar(toolbar)

        # Status bar (pack first so it sits at the very bottom)
        self._status_var = tk.StringVar(value="No file loaded. Open a FITS file to begin.")
        ttk.Label(self, textvariable=self._status_var, relief=tk.SUNKEN, anchor=tk.W,
                  padding=(4, 1)).pack(side=tk.BOTTOM, fill=tk.X)

        # Main horizontal pane: image | (fit plot + params)
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=6,
                                   sashrelief=tk.RAISED, bg="#aaa")
        main_pane.pack(fill=tk.BOTH, expand=True)

        img_frame = ttk.Frame(main_pane)
        main_pane.add(img_frame, width=560, minsize=200)
        self._build_image_panel(img_frame)

        right_pane = tk.PanedWindow(main_pane, orient=tk.VERTICAL, sashwidth=6,
                                    sashrelief=tk.RAISED, bg="#aaa")
        main_pane.add(right_pane, minsize=300)

        fit_frame = ttk.Frame(right_pane)
        right_pane.add(fit_frame, height=380, minsize=120)
        self._build_fit_panel(fit_frame)

        params_frame = ttk.Frame(right_pane)
        right_pane.add(params_frame, minsize=120)
        self._build_params_panel(params_frame)

    def _build_toolbar(self, tb):
        # File
        ttk.Button(tb, text="Open FITS", command=self._open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="Save Results", command=self._save_mode_results).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Contrast presets
        ttk.Label(tb, text="Contrast:").pack(side=tk.LEFT)
        ttk.Button(tb, text="ZScale", command=self._apply_zscale).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="1–99%", command=lambda: self._apply_percentile(1, 99)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="5–95%", command=lambda: self._apply_percentile(5, 95)).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="Min/Max", command=self._apply_minmax).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Manual vmin/vmax
        ttk.Label(tb, text="vmin:").pack(side=tk.LEFT)
        self._vmin_var = tk.StringVar(value="")
        e = ttk.Entry(tb, textvariable=self._vmin_var, width=8)
        e.pack(side=tk.LEFT, padx=2)
        e.bind("<Return>", lambda _: self._redraw_image())

        ttk.Label(tb, text="vmax:").pack(side=tk.LEFT)
        self._vmax_var = tk.StringVar(value="")
        e = ttk.Entry(tb, textvariable=self._vmax_var, width=8)
        e.pack(side=tk.LEFT, padx=2)
        e.bind("<Return>", lambda _: self._redraw_image())
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Colormap
        ttk.Label(tb, text="Cmap:").pack(side=tk.LEFT)
        self._cmap_var = tk.StringVar(value="gray")
        cb = ttk.Combobox(tb, textvariable=self._cmap_var, width=9,
                          values=["gray", "gray_r", "viridis", "plasma", "inferno",
                                  "magma", "hot", "RdBu_r"])
        cb.pack(side=tk.LEFT, padx=2)
        cb.bind("<<ComboboxSelected>>", lambda _: self._redraw_image())
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Transpose
        ttk.Button(tb, text="Flip Row/Col", command=self._transpose_data).pack(side=tk.LEFT, padx=2)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Bad pixel mode toggle
        self._bp_btn_text = tk.StringVar(value="Bad Pixel Mode: OFF")
        self._bp_btn = ttk.Button(tb, textvariable=self._bp_btn_text,
                                  command=self._toggle_bad_pixel_mode)
        self._bp_btn.pack(side=tk.LEFT, padx=4)

    def _build_image_panel(self, parent):
        ttk.Label(parent, text="Click to navigate rows  |  Bad Pixel Mode: click to toggle pixels",
                  font=("TkDefaultFont", 8), foreground="gray").pack(side=tk.TOP, anchor=tk.W, padx=4)
        fig = Figure(figsize=(5, 7), dpi=90)
        self._img_ax = fig.add_subplot(111)
        fig.tight_layout(pad=0.5)
        self._img_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._img_canvas = canvas
        canvas.mpl_connect("button_press_event", self._on_image_click)
        canvas.mpl_connect("motion_notify_event", self._on_image_motion)
        canvas.mpl_connect("key_press_event", self._on_image_key)

    def _build_fit_panel(self, parent):
        ttk.Label(parent, text="Row fit  (○ = sigma-clipped outlier  ✕ = bad pixel)",
                  font=("TkDefaultFont", 8), foreground="gray").pack(side=tk.TOP, anchor=tk.W, padx=4)
        fig = Figure(figsize=(6, 4), dpi=90)
        self._fit_ax1 = fig.add_subplot(2, 1, 1)
        self._fit_ax2 = fig.add_subplot(2, 1, 2)
        fig.tight_layout(pad=1.2)
        self._fit_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._fit_canvas = canvas
        canvas.mpl_connect("button_press_event", self._on_fit_click)
        canvas.mpl_connect("motion_notify_event", self._on_fit_motion)
        canvas.mpl_connect("key_press_event", self._on_fit_key)

    # ------------------------------------------------------------------
    # Help-button factory
    # ------------------------------------------------------------------

    def _make_help_btn(self, parent: tk.Widget, tip_text: str) -> ttk.Button:
        """Return a small '?' button that shows a dismissible popup on click."""
        btn = ttk.Button(parent, text="?", width=2)

        def _show(event=None, _btn=btn, _text=tip_text):
            popup = tk.Toplevel(self)
            popup.overrideredirect(True)
            popup.attributes("-topmost", True)
            x = _btn.winfo_rootx() + _btn.winfo_width() + 4
            y = _btn.winfo_rooty()
            lbl = tk.Label(popup, text=_text, justify=tk.LEFT,
                           bg="#fffbe6", fg="#333333",
                           relief=tk.SOLID, bd=1,
                           font=("TkDefaultFont", 10),
                           padx=6, pady=4, wraplength=320)
            lbl.pack()
            popup.geometry(f"+{x}+{y}")
            popup.update_idletasks()

            def _close(_p=popup):
                try:
                    _p.destroy()
                except tk.TclError:
                    pass

            popup.bind("<FocusOut>", lambda e: _close())
            popup.bind("<Button-1>", lambda e: _close())
            self.bind("<Button-1>", lambda e: _close(), add="+")
            popup.focus_set()

        btn.configure(command=_show)
        return btn

    def _build_params_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        nav_tab = ttk.Frame(nb, padding=6)
        nb.add(nav_tab, text="Navigate & Fit")
        self._build_nav_tab(nav_tab)

        common_tab = ttk.Frame(nb, padding=6)
        nb.add(common_tab, text="Common")
        self._build_common_tab(common_tab)

        fit_tab = ttk.Frame(nb, padding=6)
        nb.add(fit_tab, text="Background")
        self._build_fit_params_tab(fit_tab)

        mask_tab = ttk.Frame(nb, padding=6)
        nb.add(mask_tab, text="Bad Pixels")
        self._build_mask_tab(mask_tab)

        extr_tab = ttk.Frame(nb, padding=6)
        nb.add(extr_tab, text="Extract 1D")
        self._build_extract_tab(extr_tab)

    def _build_nav_tab(self, parent):
        # Row navigation
        nav = ttk.LabelFrame(parent, text="Row Navigation", padding=4)
        nav.pack(fill=tk.X, pady=(0, 4))

        # Row 0 — standard prev/next + row entry
        nav_r0 = ttk.Frame(nav)
        nav_r0.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(nav_r0, text="◀ Prev", command=self._prev_row).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_r0, text="Next ▶", command=self._next_row).pack(side=tk.LEFT, padx=2)
        ttk.Label(nav_r0, text="  Row:").pack(side=tk.LEFT)
        self._row_var = tk.IntVar(value=0)
        e = ttk.Entry(nav_r0, textvariable=self._row_var, width=6)
        e.pack(side=tk.LEFT, padx=2)
        e.bind("<Return>", lambda _: self._go_to_row(self._row_var.get()))
        self._n_rows_label = ttk.Label(nav_r0, text="/ 0")
        self._n_rows_label.pack(side=tk.LEFT)

        # Row 1 — failed-row navigation + flag toggle
        nav_r1 = ttk.Frame(nav)
        nav_r1.pack(fill=tk.X)
        ttk.Button(nav_r1, text="⏮ Prev Failed",
                   command=self._prev_failed_row).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_r1, text="Next Failed ⏭",
                   command=self._next_failed_row).pack(side=tk.LEFT, padx=2)
        ttk.Separator(nav_r1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._flag_btn_var = tk.StringVar(value="⚑ Flag Row")
        ttk.Button(nav_r1, textvariable=self._flag_btn_var,
                   command=self._toggle_row_flag).pack(side=tk.LEFT, padx=2)
        self._flag_status_var = tk.StringVar(value="")
        ttk.Label(nav_r1, textvariable=self._flag_status_var,
                  foreground="orange", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=4)

        # Fit mode selector
        mode_lf = ttk.LabelFrame(parent, text="Fit Mode", padding=4)
        mode_lf.pack(fill=tk.X, pady=(0, 4))
        self._fit_mode_var = tk.StringVar(value="background")
        rb1 = ttk.Radiobutton(mode_lf, text="Background (Chebyshev)",
                               variable=self._fit_mode_var, value="background")
        rb1.pack(side=tk.LEFT, padx=4)
        rb2 = ttk.Radiobutton(mode_lf, text="Profile (Gauss-Hermite)",
                               variable=self._fit_mode_var, value="profile")
        rb2.pack(side=tk.LEFT, padx=4)
        self._mode_radios = [rb1, rb2]

        # Actions — grid layout, one logical group per row
        act = ttk.LabelFrame(parent, text="Actions", padding=6)
        act.pack(fill=tk.X, pady=(0, 4))

        # Row 0 — single-row and display
        r0 = ttk.Frame(act)
        r0.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(r0, text="Fit Current Row",
                   command=self._fit_mode_current_row).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(r0, text="Display",
                   command=self._display_stored_spectrum).pack(side=tk.LEFT, padx=4)

        # Row 1 — batch loop controls
        r1 = ttk.Frame(act)
        r1.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(r1, text="Fit All Rows",
                   command=self._fit_mode_start_all).pack(side=tk.LEFT, padx=(0, 4))
        self._stop_btn = ttk.Button(r1, text="Stop",
                                    command=self._fit_mode_stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(r1, text="Resume",
                   command=self._fit_mode_resume).pack(side=tk.LEFT, padx=4)

        # Row 1b — profile diagnostics
        r1b = ttk.Frame(act)
        r1b.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(r1b, text="Centroid Plot",
                   command=self._show_centroid_plot).pack(side=tk.LEFT, padx=(0, 4))
        self._make_help_btn(r1b,
            "Plot profile centroid position vs row.\n"
            "Click any point in the plot to navigate to that row.").pack(side=tk.LEFT)

        # Row 2 — options / toggles
        r2 = ttk.Frame(act)
        r2.pack(fill=tk.X)
        self._autofit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="Auto-fit on navigate",
                        variable=self._autofit_var).pack(side=tk.LEFT, padx=(0, 12))
        self._live_display_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="Live display during Fit All",
                        variable=self._live_display_var).pack(side=tk.LEFT)

        # Progress
        self._progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(parent, variable=self._progress_var, maximum=100).pack(fill=tk.X, pady=(0, 4))

        # Danger zone — separated visually
        danger_lf = ttk.LabelFrame(parent, text="Reset", padding=6)
        danger_lf.pack(fill=tk.X, pady=(4, 4))
        ttk.Button(danger_lf, text="Restart (clear all results)",
                   command=self._restart_results).pack(anchor=tk.W)

        # Keyboard shortcuts hint
        shortcuts_row = ttk.Frame(parent)
        shortcuts_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(shortcuts_row, text="Keyboard shortcuts").pack(side=tk.LEFT, padx=(0, 4))
        self._make_help_btn(shortcuts_row,
            "← / → or ↑ / ↓   navigate rows\n"
            "F   fit current row\n"
            "B   toggle bad pixel mode\n"
            "D   toggle bad pixel under cursor").pack(side=tk.LEFT)

    def _build_common_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        r = 0

        # --- Fit Region ---
        region_lf = ttk.LabelFrame(parent, text="Fit Region", padding=4)
        region_lf.grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=(0, 6))
        region_lf.columnconfigure(1, weight=1)
        r += 1

        # Region mode row
        mode_row = ttk.Frame(region_lf)
        mode_row.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(mode_row, text="Mode:").pack(side=tk.LEFT)
        self._mode_var = tk.StringVar(value="full")
        self._mode_full_rb = ttk.Radiobutton(mode_row, text="Full", variable=self._mode_var, value="full")
        self._mode_full_rb.pack(side=tk.LEFT, padx=(4, 0))
        self._mode_trace_rb = ttk.Radiobutton(mode_row, text="Trace", variable=self._mode_var, value="trace")
        self._mode_trace_rb.pack(side=tk.LEFT)
        self._make_help_btn(mode_row,
            "Full: regions are absolute column indices.\n"
            "Trace: the bounding box of all regions is interpreted relative to the traced centre column.").pack(side=tk.LEFT, padx=(4, 0))

        # Regions listbox
        lb_frame = ttk.Frame(region_lf)
        lb_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(4, 2))
        lb_frame.columnconfigure(0, weight=1)
        self._regions_listbox = tk.Listbox(lb_frame, height=4, selectmode=tk.SINGLE,
                                            font=("TkDefaultFont", 9))
        self._regions_listbox.grid(row=0, column=0, sticky=tk.EW)
        lb_scroll = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL,
                                   command=self._regions_listbox.yview)
        lb_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._regions_listbox.config(yscrollcommand=lb_scroll.set)

        # Manual entry row
        entry_row = ttk.Frame(region_lf)
        entry_row.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        self._region_entry_var = tk.StringVar(value="")
        ttk.Entry(entry_row, textvariable=self._region_entry_var, width=10).pack(side=tk.LEFT)
        ttk.Button(entry_row, text="Add", command=self._add_region_manual).pack(
            side=tk.LEFT, padx=(4, 0))
        self._make_help_btn(entry_row,
            "Type a region as  start-stop  (e.g. 100-250 or 100 250)\n"
            "then click Add to append it to the list.").pack(side=tk.LEFT, padx=(4, 0))

        # Select region on image button
        sel_row = ttk.Frame(region_lf)
        sel_row.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 2))
        self._region_click_btn = ttk.Button(
            sel_row, text="Select region on image",
            command=self._start_region_click)
        self._region_click_btn.pack(side=tk.LEFT)
        self._region_click_status_var = tk.StringVar(value="")
        ttk.Label(sel_row, textvariable=self._region_click_status_var,
                  font=("TkDefaultFont", 8), foreground="gray").pack(side=tk.LEFT, padx=(6, 0))
        self._make_help_btn(sel_row,
            "Click this button, then click two points on the 2D image to add a region.\n\n"
            "Background mode: adds a new fit region.\n"
            "Profile mode: sets the extraction aperture low/high bounds and switches to Custom bounds.\n"
            "Right-click on the image to remove the nearest background region.").pack(side=tk.LEFT, padx=(4, 0))

        # Remove / Clear buttons
        rmv_row = ttk.Frame(region_lf)
        rmv_row.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Button(rmv_row, text="Remove selected",
                   command=self._remove_selected_region).pack(side=tk.LEFT)
        ttk.Button(rmv_row, text="Clear all",
                   command=lambda: (self._fit_regions.clear(),
                                    self._refresh_regions_listbox(),
                                    self._draw_extr_region_lines())).pack(side=tk.LEFT, padx=(6, 0))

        # Offset row
        offset_row = ttk.Frame(region_lf)
        offset_row.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(offset_row, text="Offset (trace):").pack(side=tk.LEFT)
        self._offset_var = tk.IntVar(value=0)
        ttk.Spinbox(offset_row, from_=-9999, to=9999, textvariable=self._offset_var, width=8).pack(
            side=tk.LEFT, padx=4)
        self._make_help_btn(offset_row,
            "Shift the trace centre by this many columns before computing the fit window.\n"
            "Only active in Trace mode.").pack(side=tk.LEFT)

        trace_lf = ttk.LabelFrame(region_lf, text="Trace Definition  (trace mode only)", padding=4)
        trace_lf.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=(6, 2))
        trace_lf.columnconfigure(1, weight=1)

        coeff_row = ttk.Frame(trace_lf)
        coeff_row.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        ttk.Label(coeff_row, text="Poly coeffs:",
                  font=("TkDefaultFont", 8)).pack(side=tk.LEFT)
        self._trace_coeffs_var = tk.StringVar(value="")
        ttk.Entry(coeff_row, textvariable=self._trace_coeffs_var, width=20).pack(
            side=tk.LEFT, padx=4)
        self._make_help_btn(coeff_row,
            "Polynomial coefficients in np.polyval order (highest power first).\n\n"
            'Examples:\n  "512"        → constant centre at column 512\n'
            '  "0.05, 512"  → centre = 0.05·row + 512 (linear tilt)').pack(side=tk.LEFT)

        ttk.Button(trace_lf, text="Apply Trace",
                   command=self._apply_trace_coeffs).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        self._trace_status_var = tk.StringVar(value="No trace defined.")
        ttk.Label(trace_lf, textvariable=self._trace_status_var,
                  font=("TkDefaultFont", 8), foreground="gray").grid(
            row=2, column=0, columnspan=2, sticky=tk.W)

        # --- Sigma Clipping ---
        clip_lf = ttk.LabelFrame(parent, text="Sigma Clipping  (background & profile)", padding=4)
        clip_lf.grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        clip_lf.columnconfigure(1, weight=1)
        r += 1

        clip_lf.columnconfigure(2, weight=1)
        self._sigma_clip_var = tk.BooleanVar(value=True)
        clip_en_row = ttk.Frame(clip_lf)
        clip_en_row.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=2)
        ttk.Checkbutton(clip_en_row, text="Enable", variable=self._sigma_clip_var).pack(side=tk.LEFT)
        self._make_help_btn(clip_en_row,
            "Enable iterative sigma clipping during fitting.\n\n"
            "Background mode: clips residuals above/below sigma thresholds.\n"
            "Profile mode: uses the Sigma upper value as a symmetric rejection threshold.").pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(clip_lf, text="Sigma upper:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self._sigma_upper_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(clip_lf, from_=0.5, to=20.0, increment=0.5,
                    textvariable=self._sigma_upper_var, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=4)
        self._make_help_btn(clip_lf,
            "Reject pixels more than this many standard deviations above the fit.\n"
            "Also used as the symmetric threshold in Profile mode.").grid(
            row=1, column=2, sticky=tk.W)

        ttk.Label(clip_lf, text="Sigma lower:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self._sigma_lower_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(clip_lf, from_=0.5, to=20.0, increment=0.5,
                    textvariable=self._sigma_lower_var, width=8).grid(
            row=2, column=1, sticky=tk.W, padx=4)
        self._make_help_btn(clip_lf,
            "Reject pixels more than this many standard deviations below the fit.\n"
            "(Background mode only; Profile mode uses Sigma upper symmetrically.)").grid(
            row=2, column=2, sticky=tk.W)

        ttk.Label(clip_lf, text="Max iterations:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self._max_iters_var = tk.IntVar(value=3)
        ttk.Spinbox(clip_lf, from_=1, to=50, textvariable=self._max_iters_var, width=8).grid(
            row=3, column=1, sticky=tk.W, padx=4)
        self._make_help_btn(clip_lf,
            "Maximum number of sigma-clipping iterations per row.\n"
            "Fitting stops early if no new pixels are rejected.").grid(
            row=3, column=2, sticky=tk.W)

        # Connect shared vars to the mid-fit change handler
        for var in (self._offset_var,
                    self._sigma_upper_var, self._sigma_lower_var,
                    self._max_iters_var, self._sigma_clip_var):
            var.trace_add("write", self._on_param_change)

        # Redraw region lines when offset changes
        self._offset_var.trace_add("write", lambda *_: self._draw_extr_region_lines())

    # ------------------------------------------------------------------
    # Fit-region list helpers
    # ------------------------------------------------------------------

    def _refresh_regions_listbox(self):
        self._regions_listbox.delete(0, tk.END)
        for lo, hi in self._fit_regions:
            self._regions_listbox.insert(tk.END, f"col {lo} – {hi}")

    def _add_region_manual(self):
        import re
        text = self._region_entry_var.get().strip()
        m = re.match(r"(\d+)\s*[-,\s]\s*(\d+)", text)
        if not m:
            self._status_var.set("Enter region as  start-stop  e.g.  100-250")
            return
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo >= hi:
            lo, hi = hi, lo
        self._fit_regions.append((lo, hi))
        self._fit_regions.sort()
        self._region_entry_var.set("")
        self._refresh_regions_listbox()
        self._draw_extr_region_lines()
        self._status_var.set(f"Added region col {lo} – {hi}.")

    def _remove_selected_region(self):
        sel = self._regions_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        lo, hi = self._fit_regions.pop(idx)
        self._refresh_regions_listbox()
        self._draw_extr_region_lines()
        self._status_var.set(f"Removed region col {lo} – {hi}.")

    def _remove_region_at_col(self, col: int):
        if not self._fit_regions:
            return
        # prefer a region that contains the click column
        for i, (lo, hi) in enumerate(self._fit_regions):
            if lo <= col <= hi:
                self._fit_regions.pop(i)
                self._refresh_regions_listbox()
                self._draw_extr_region_lines()
                self._status_var.set(f"Removed region col {lo} – {hi}.")
                return
        # else remove the region with the nearest boundary
        nearest = min(range(len(self._fit_regions)),
                      key=lambda i: min(abs(col - self._fit_regions[i][0]),
                                        abs(col - self._fit_regions[i][1])))
        lo, hi = self._fit_regions.pop(nearest)
        self._refresh_regions_listbox()
        self._draw_extr_region_lines()
        self._status_var.set(f"Removed region col {lo} – {hi}.")

    def _build_fit_params_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=0)

        ttk.Label(parent,
                  text="Chebyshev polynomial background subtraction.\n"
                       "Fits a polynomial to unmasked columns in each row,\n"
                       "then subtracts it to produce the background-subtracted 2D spectrum.",
                  font=("TkDefaultFont", 10), foreground="#555555").grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(2, 10))

        deg_row = ttk.Frame(parent)
        deg_row.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)
        ttk.Label(deg_row, text="Degree:", font=("TkDefaultFont", 10)).pack(side=tk.LEFT)
        self._degree_var = tk.IntVar(value=3)
        ttk.Spinbox(deg_row, from_=1, to=30, textvariable=self._degree_var, width=8).pack(
            side=tk.LEFT, padx=4)
        self._make_help_btn(deg_row,
            "Degree of the Chebyshev polynomial fitted to background columns.\n\n"
            "Low values (1–3) give smooth, broad backgrounds.\n"
            "Higher values track narrower features but risk overfitting.").pack(side=tk.LEFT)
        self._degree_var.trace_add("write", self._on_param_change)

    def _build_mask_tab(self, parent):
        self._bp_status_var = tk.StringVar(value="Bad pixel mode: OFF")
        ttk.Label(parent, textvariable=self._bp_status_var, foreground="gray").pack(anchor=tk.W)
        ttk.Button(parent, text="Toggle Bad Pixel Mode (B)",
                   command=self._toggle_bad_pixel_mode).pack(anchor=tk.W, pady=4)

        bp_info_row = ttk.Frame(parent)
        bp_info_row.pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(bp_info_row, text="Bad Pixel Mode").pack(side=tk.LEFT)
        self._make_help_btn(bp_info_row,
            "Activate Bad Pixel Mode (button above or press B), then click pixels on the image "
            "or the row-fit plot to toggle them as bad.\n\n"
            "Bad pixels are excluded from all fits and shown in red.\n"
            "Press D to toggle the pixel under the cursor without entering full bad-pixel mode.").pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Button(parent, text="Clear Manual Mask — Current Row",
                   command=self._clear_row_mask).pack(anchor=tk.W, pady=2)
        ttk.Button(parent, text="Clear All Manual Masks",
                   command=self._clear_all_masks).pack(anchor=tk.W, pady=2)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(parent, text="Load file mask from FITS extension:").pack(anchor=tk.W)
        mf = ttk.Frame(parent)
        mf.pack(anchor=tk.W, pady=2)
        self._mask_ext_var = tk.IntVar(value=1)
        ttk.Spinbox(mf, from_=0, to=99, textvariable=self._mask_ext_var, width=5).pack(side=tk.LEFT)
        ttk.Button(mf, text="Load", command=self._load_mask_ext).pack(side=tk.LEFT, padx=4)

        # ---- Cosmic Ray Removal (lacosmic) --------------------------------
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        cr_lf = ttk.LabelFrame(parent, text="Cosmic Ray Removal (lacosmic)", padding=4)
        cr_lf.pack(fill=tk.X, pady=(0, 4))

        if not HAS_LACOSMIC:
            ttk.Label(cr_lf, text="astroscrappy not installed.\npip install astroscrappy",
                      foreground="gray", font=("TkDefaultFont", 8)).pack(anchor=tk.W)
        else:
            def _row(label, var, from_, to_, width=7):
                f = ttk.Frame(cr_lf)
                f.pack(fill=tk.X, pady=1)
                ttk.Label(f, text=label, width=14, anchor=tk.W).pack(side=tk.LEFT)
                ttk.Spinbox(f, from_=from_, to=to_, increment=0.1,
                            textvariable=var, width=width).pack(side=tk.LEFT)

            self._cr_sigclip_var  = tk.DoubleVar(value=4.5)
            self._cr_sigfrac_var  = tk.DoubleVar(value=0.3)
            self._cr_objlim_var   = tk.DoubleVar(value=5.0)
            self._cr_gain_var     = tk.DoubleVar(value=1.0)
            self._cr_readnoise_var = tk.DoubleVar(value=6.5)
            self._cr_maxiter_var  = tk.IntVar(value=4)

            _row("sigclip",   self._cr_sigclip_var,   0.1, 20.0)
            _row("sigfrac",   self._cr_sigfrac_var,   0.0,  1.0)
            _row("objlim",    self._cr_objlim_var,    0.1, 50.0)
            _row("gain",      self._cr_gain_var,      0.1, 20.0)
            _row("readnoise", self._cr_readnoise_var, 0.0, 100.0)

            fi = ttk.Frame(cr_lf)
            fi.pack(fill=tk.X, pady=1)
            ttk.Label(fi, text="maxiter", width=14, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Spinbox(fi, from_=1, to=20, increment=1,
                        textvariable=self._cr_maxiter_var, width=7).pack(side=tk.LEFT)

            bf = ttk.Frame(cr_lf)
            bf.pack(fill=tk.X, pady=(4, 0))
            ttk.Button(bf, text="Run lacosmic", command=self._run_lacosmic).pack(side=tk.LEFT, padx=(0, 4))
            ttk.Button(bf, text="Clear CR Mask", command=self._clear_cr_mask).pack(side=tk.LEFT)

            peek_btn = tk.Button(cr_lf, text="Hold to hide CR mask",
                                 relief=tk.RAISED, bd=2)
            peek_btn.pack(anchor=tk.W, pady=(4, 0))
            peek_btn.bind("<ButtonPress-1>",   lambda _e: self._peek_cr_press())
            peek_btn.bind("<ButtonRelease-1>", lambda _e: self._peek_cr_release())

            self._cr_status_var = tk.StringVar(value="No CR mask.")
            ttk.Label(cr_lf, textvariable=self._cr_status_var,
                      font=("TkDefaultFont", 8), foreground="gray").pack(anchor=tk.W, pady=(2, 0))

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self):
        self.bind("<Left>",  lambda _: self._prev_row())
        self.bind("<Right>", lambda _: self._next_row())
        self.bind("<Up>",    lambda _: self._prev_row())
        self.bind("<Down>",  lambda _: self._next_row())
        self.bind("b", lambda _: self._toggle_bad_pixel_mode())
        self.bind("f", lambda _: self._fit_mode_current_row())

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _open_file(self):
        if not HAS_FITS:
            messagebox.showerror("Missing dependency", "astropy is required to read FITS files.")
            return
        path = filedialog.askopenfilename(
            title="Open FITS file",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")],
        )
        if path:
            self._load_fits(path)

    def _load_fits(self, path: str):
        if not HAS_FITS:
            messagebox.showerror("Missing dependency", "astropy is required.")
            return
        try:
            with _fits.open(path) as hdul:
                # Collect extensions that contain 2-D image data
                image_exts = [
                    i for i, hdu in enumerate(hdul)
                    if hdu.data is not None and np.asarray(hdu.data).ndim == 2
                ]
                if len(image_exts) == 0:
                    messagebox.showerror("Error", "No 2D image data found in this FITS file.")
                    return
                if len(image_exts) == 1:
                    idx = image_exts[0]
                else:
                    # Build human-readable labels for each candidate extension
                    labels = []
                    for i in image_exts:
                        hdu = hdul[i]
                        name = hdu.name or f"HDU {i}"
                        shape = np.asarray(hdu.data).shape
                        labels.append(f"[{i}]  {name}  {shape[0]}×{shape[1]}")
                    idx = _ask_extension(self, labels, image_exts)
                    if idx is None:
                        return  # user cancelled
                data = np.asarray(hdul[idx].data, dtype=float)
                self.fits_header = hdul[idx].header
        except Exception as e:
            messagebox.showerror("Read error", str(e))
            return
        if data.ndim != 2:
            messagebox.showerror("Error", f"Expected 2D FITS data, got shape {data.shape}.")
            return

        self.fits_path = path
        self.trace = data
        n_rows, n_cols = data.shape
        self.manual_mask = np.zeros((n_rows, n_cols), dtype=bool)
        self.file_mask = np.zeros((n_rows, n_cols), dtype=bool)
        self.cr_mask = None
        self.cr_unmask = None
        self.clip_mask = None
        self.row_flag = np.zeros(n_rows, dtype=bool)
        self.results = [None] * n_rows
        self.centers = None
        self._trace_line = None
        self.ivar = None
        self._var_file_var.set("")
        self._var_status_var.set("No variance loaded — uniform weights used.")
        self.flux_1d = None
        self.err_1d = None
        self.current_row = 0
        self._region_click_state = 0
        self._extr_vlines = []
        self._trace_status_var.set("No trace defined.")
        self._mode_var.set("full")

        self._fit_regions = []
        self._refresh_regions_listbox()
        self._region_click_lo = None
        self._n_rows_label.config(text=f"/ {n_rows - 1}")
        self.title(f"2D Spectrum Fitter — {os.path.basename(path)}")
        self._status_var.set(f"Loaded: {path}  |  {n_rows} rows × {n_cols} cols")

        self._apply_zscale()       # sets vmin/vmax and calls _draw_image
        self._go_to_row(0)

    def _transpose_data(self):
        if self.trace is None:
            return
        self.trace = self.trace.T
        for attr in ("manual_mask", "file_mask", "cr_mask", "cr_unmask", "ivar"):
            arr = getattr(self, attr)
            if arr is not None:
                setattr(self, attr, arr.T)
        # clip_mask and centers are row-indexed; reset them
        self.clip_mask = None
        self.centers = None
        self._trace_line = None
        self._trace_status_var.set("No trace defined.")
        n_rows, n_cols = self.trace.shape
        self.row_flag = np.zeros(n_rows, dtype=bool)
        self.results = [None] * n_rows
        self.profile_fit_cache = None
        self.flux_1d = None
        self.err_1d = None
        self.current_row = 0
        self._fit_regions = []
        self._refresh_regions_listbox()
        self._n_rows_label.config(text=f"/ {n_rows - 1}")
        self._status_var.set(
            f"Transposed  |  {n_rows} rows × {n_cols} cols"
            + (f"  —  {os.path.basename(self.fits_path)}" if self.fits_path else "")
        )
        self._redraw_mask_overlay()
        self._apply_zscale()
        self._go_to_row(0)

    def _restart_results(self):
        if self.trace is None:
            return
        has_bkg = any(r is not None for r in self.results)
        has_1d = self.flux_1d is not None and np.any(np.isfinite(self.flux_1d))
        if has_bkg or has_1d:
            what = []
            if has_bkg:
                n = sum(1 for r in self.results if r is not None)
                what.append(f"{n} background-fitted rows")
            if has_1d:
                n = int(np.sum(np.isfinite(self.flux_1d)))
                what.append(f"{n} extracted 1D rows")
            if not messagebox.askyesno(
                "Clear all results",
                "This will erase:\n  • " + "\n  • ".join(what) + "\n\nContinue?",
            ):
                return
        n_rows = self.trace.shape[0]
        self.results = [None] * n_rows
        self.flux_1d = np.full(n_rows, np.nan)
        self.err_1d = np.full(n_rows, np.nan)
        self.clip_mask = None
        self.row_flag = np.zeros(n_rows, dtype=bool)
        self._fit_row_idx = 0
        self._extr_row_idx = 0
        self._redraw_mask_overlay()
        self._update_flag_overlay()
        self._update_flag_btn()
        self._status_var.set("All results cleared.")

    def _confirm_incomplete_save(self, n_done: int, n_total: int, missing: list) -> bool:
        preview = ", ".join(str(i) for i in missing[:5])
        if len(missing) > 5:
            preview += f" … (+{len(missing) - 5} more)"
        return messagebox.askyesno(
            "Incomplete fit",
            f"Only {n_done} of {n_total} rows have been fitted.\n"
            f"Unfitted rows: {preview}\n"
            "These rows will be NaN in the output.\nSave anyway?",
        )

    def _save_mode_results(self):
        if self._fit_mode_var is not None and self._fit_mode_var.get() == "profile":
            self._save_1d_fits()
        else:
            self._save_results()

    def _save_results(self):
        if self.trace is None or self.fits_path is None:
            messagebox.showwarning("No data", "Load a FITS file first.")
            return
        n_fitted = sum(1 for r in self.results if r is not None and r.get("model") is not None)
        if n_fitted == 0:
            messagebox.showwarning("No results", "Run background fitting first.")
            return
        if not HAS_FITS:
            messagebox.showerror("Missing dependency", "astropy is required to save FITS files.")
            return
        n_rows, n_cols = self.trace.shape
        missing = [i for i, r in enumerate(self.results)
                   if r is None or r.get("model") is None]
        if n_fitted < n_rows:
            unflagged = [i for i in missing
                         if self.row_flag is None or not self.row_flag[i]]
            if unflagged:
                if not self._confirm_incomplete_save(n_fitted, n_rows, unflagged):
                    return
        if missing and self.row_flag is not None:
            for i in missing:
                self.row_flag[i] = True
            self._update_flag_btn()
            self._update_flag_overlay()
        resid_rows = []
        for i, r in enumerate(self.results):
            if r is None or r.get("model") is None:
                resid_rows.append(np.full(n_cols, np.nan))
            else:
                resid_rows.append(self.trace[i, :] - r["model"])
        resid_arr = np.vstack(resid_rows)
        base = os.path.splitext(os.path.basename(self.fits_path))[0]
        out_dir = os.path.dirname(self.fits_path)
        path = filedialog.asksaveasfilename(
            title="Save background-subtracted 2D spectrum",
            initialdir=out_dir,
            initialfile=f"{base}_bkgsub.fits",
            defaultextension=".fits",
            filetypes=[("FITS files", "*.fits *.fit"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            hdu0 = _fits.PrimaryHDU(data=resid_arr, header=self.fits_header)
            _fits.HDUList([hdu0]).writeto(path, overwrite=True)
            self._status_var.set(f"Saved: {path}")
            messagebox.showinfo("Saved", f"Background-subtracted spectrum written to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _load_mask_ext(self):
        if self.fits_path is None or self.trace is None:
            messagebox.showwarning("No file", "Load a FITS file first.")
            return
        ext = self._mask_ext_var.get()
        try:
            with _fits.open(self.fits_path) as hdul:
                if ext >= len(hdul):
                    messagebox.showerror("Error", f"Extension {ext} not found (file has {len(hdul)} ext).")
                    return
                mask_data = np.asarray(hdul[ext].data, dtype=bool)
        except Exception as e:
            messagebox.showerror("Read error", str(e))
            return
        if mask_data.shape != self.trace.shape:
            messagebox.showerror(
                "Shape mismatch",
                f"Mask shape {mask_data.shape} does not match data shape {self.trace.shape}.",
            )
            return
        self.file_mask = mask_data
        self._redraw_mask_overlay()
        self._redraw_current_row()
        self._status_var.set(f"Loaded mask from extension {ext}  ({mask_data.sum()} bad pixels)")

    def _run_lacosmic(self):
        if self.trace is None:
            messagebox.showwarning("No data", "Load a FITS file first.")
            return
        if not HAS_LACOSMIC:
            messagebox.showerror("Missing dependency", "astroscrappy is required for cosmic ray removal.")
            return
        self._status_var.set("Running lacosmic …")
        self.update_idletasks()
        try:
            crmask, _ = astroscrappy.detect_cosmics(
                self.trace.astype(np.float32),
                sigclip=float(self._cr_sigclip_var.get()),
                sigfrac=float(self._cr_sigfrac_var.get()),
                objlim=float(self._cr_objlim_var.get()),
                gain=float(self._cr_gain_var.get()),
                readnoise=float(self._cr_readnoise_var.get()),
                niter=int(self._cr_maxiter_var.get()),
                verbose=False,
            )
        except Exception as e:
            messagebox.showerror("lacosmic error", str(e))
            self._status_var.set("lacosmic failed.")
            return
        self.cr_mask = crmask.astype(bool)
        n_cr = int(self.cr_mask.sum())
        self._cr_status_var.set(f"CR mask: {n_cr} pixels flagged.")
        self._status_var.set(f"lacosmic done — {n_cr} cosmic ray pixels flagged.")
        self._invalidate_all_fits()
        self._redraw_mask_overlay()
        self._redraw_current_row()

    def _clear_cr_mask(self):
        if self.trace is None:
            return
        self.cr_mask = None
        self.cr_unmask = None
        self._cr_status_var.set("No CR mask.")
        self._status_var.set("Cosmic ray mask cleared.")
        self._invalidate_all_fits()
        self._redraw_mask_overlay()
        self._redraw_current_row()

    def _invalidate_all_fits(self):
        """Discard all stored fit results so rows are re-fitted with the current mask."""
        if self.results:
            self.results = [None] * len(self.results)
        if self.profile_fit_cache is not None:
            self.profile_fit_cache = [None] * len(self.profile_fit_cache)

    # ------------------------------------------------------------------
    # Contrast controls
    # ------------------------------------------------------------------

    def _apply_zscale(self):
        if self.trace is None:
            return
        vmin, vmax = _zscale(self.trace)
        self._vmin_var.set(f"{vmin:.5g}")
        self._vmax_var.set(f"{vmax:.5g}")
        self._draw_image()

    def _apply_percentile(self, lo: float, hi: float):
        if self.trace is None:
            return
        valid = self.trace[np.isfinite(self.trace)]
        self._vmin_var.set(f"{np.percentile(valid, lo):.5g}")
        self._vmax_var.set(f"{np.percentile(valid, hi):.5g}")
        self._redraw_image()

    def _apply_minmax(self):
        if self.trace is None:
            return
        valid = self.trace[np.isfinite(self.trace)]
        self._vmin_var.set(f"{valid.min():.5g}")
        self._vmax_var.set(f"{valid.max():.5g}")
        self._redraw_image()

    def _get_vmin_vmax(self):
        try:
            return float(self._vmin_var.get()), float(self._vmax_var.get())
        except ValueError:
            if self.trace is not None:
                return _zscale(self.trace)
            return 0.0, 1.0

    # ------------------------------------------------------------------
    # Image drawing
    # ------------------------------------------------------------------

    def _draw_image(self):
        if self.trace is None:
            return
        vmin, vmax = self._get_vmin_vmax()
        ax = self._img_ax
        ax.clear()
        self._mask_scatter = None
        self._img_im = ax.imshow(
            self.trace, origin="lower", aspect="auto",
            vmin=vmin, vmax=vmax, cmap=self._cmap_var.get(),
            interpolation="nearest",
        )
        self._row_line = ax.axhline(self.current_row, color="lime", lw=1.0, alpha=0.9)
        ax.set_xlabel("Column", fontsize=8)
        ax.set_ylabel("Row", fontsize=8)
        ax.tick_params(labelsize=7)
        self._extr_vlines = []   # cleared by ax.clear()
        self._img_fig.tight_layout(pad=0.5)
        self._redraw_mask_overlay(redraw=False)
        self._draw_trace_overlay(redraw=False)
        self._draw_extr_region_lines(redraw=False)
        self._img_canvas.draw()

    def _redraw_image(self):
        if self.trace is None:
            return
        if self._img_im is None:
            self._draw_image()
            return
        vmin, vmax = self._get_vmin_vmax()
        self._img_im.set_clim(vmin, vmax)
        self._img_im.set_cmap(self._cmap_var.get())
        self._img_canvas.draw_idle()

    def _redraw_mask_overlay(self, redraw: bool = True):
        if self.trace is None:
            return
        if self._mask_scatter is not None:
            try:
                self._mask_scatter.remove()
            except Exception:
                pass
            self._mask_scatter = None
        combined = self._display_mask()
        if combined.any():
            rows_b, cols_b = np.where(combined)
            self._mask_scatter = self._img_ax.scatter(
                cols_b, rows_b, c="red", s=2, marker="s", alpha=0.6, linewidths=0, zorder=3,
            )
        if redraw:
            self._img_canvas.draw_idle()

    def _update_row_line(self):
        if self._row_line is not None:
            self._row_line.set_ydata([self.current_row, self.current_row])
        self._draw_extr_region_lines()   # re-draws at new row (matters in trace mode)

    def _combined_mask(self) -> np.ndarray:
        """Full mask used for fitting: manual + file + CR + sigma-clipped profile pixels."""
        if self.trace is None:
            return np.zeros((0, 0), dtype=bool)
        m = self.file_mask | self.manual_mask
        if self.cr_mask is not None:
            cr = self.cr_mask.copy()
            if self.cr_unmask is not None:
                cr = cr & ~self.cr_unmask
            m = m | cr
        if self.clip_mask is not None:
            m = m | self.clip_mask
        return m

    def _display_mask(self) -> np.ndarray:
        """Like _combined_mask but hides CR layer when 'Hold to peek' is active."""
        if self.trace is None:
            return np.zeros((0, 0), dtype=bool)
        m = self.file_mask | self.manual_mask
        if not self._cr_hidden and self.cr_mask is not None:
            cr = self.cr_mask.copy()
            if self.cr_unmask is not None:
                cr = cr & ~self.cr_unmask
            m = m | cr
        if self.clip_mask is not None:
            m = m | self.clip_mask
        return m

    # ------------------------------------------------------------------
    # Row navigation
    # ------------------------------------------------------------------

    def _go_to_row(self, row: int):
        if self.trace is None:
            return
        row = int(np.clip(row, 0, self.trace.shape[0] - 1))
        self.current_row = row
        self._row_var.set(row)
        self._update_row_line()
        self._update_flag_btn()
        if self._autofit_var.get():
            self._fit_mode_current_row(update_image=False)
        else:
            self._redraw_current_row()

    def _prev_row(self):
        self._go_to_row(self.current_row - 1)

    def _next_row(self):
        self._go_to_row(self.current_row + 1)

    def _get_failed_rows(self) -> list:
        """Return row indices where the current mode's fit result is missing/NaN."""
        if self.trace is None:
            return []
        n_rows = self.trace.shape[0]
        if self._fit_mode_var is not None and self._fit_mode_var.get() == "profile":
            if self.flux_1d is None:
                return list(range(n_rows))
            return [i for i in range(n_rows) if not np.isfinite(self.flux_1d[i])]
        else:
            return [i for i, r in enumerate(self.results)
                    if r is None or r.get("model") is None]

    def _prev_failed_row(self):
        failed = self._get_failed_rows()
        candidates = [i for i in failed if i < self.current_row]
        if candidates:
            self._go_to_row(candidates[-1])
        else:
            self._status_var.set("No failed row before current position.")

    def _next_failed_row(self):
        failed = self._get_failed_rows()
        candidates = [i for i in failed if i > self.current_row]
        if candidates:
            self._go_to_row(candidates[0])
        else:
            self._status_var.set("No failed row after current position.")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _get_bkg_regions(self, row_idx: int) -> list:
        """Return validated list of (start_col, stop_col) for background fitting.
        Falls back to the full row when no regions are defined."""
        if self.trace is None:
            return []
        n_cols = self.trace.shape[1]
        if not self._fit_regions:
            return [(0, n_cols)]
        result = []
        for lo, hi in self._fit_regions:
            s = int(np.clip(lo, 0, n_cols - 2))
            e = int(np.clip(hi, s + 1, n_cols))
            result.append((s, e))
        return result

    def _get_fit_region(self, row_idx: int):
        """Single (start, stop) for profile extraction aperture — bounding box of fit regions."""
        n_cols = self.trace.shape[1]
        if self._fit_regions:
            lo = min(s for s, e in self._fit_regions)
            hi = max(e for s, e in self._fit_regions)
        else:
            lo, hi = 0, n_cols
        if self._mode_var.get() == "trace" and self.centers is not None:
            center = int(round(float(self.centers[row_idx]) + self._offset_var.get()))
            half = max(1, (hi - lo) // 2)
            lo = int(np.clip(center - half, 0, n_cols - 1))
            hi = int(np.clip(center + half + 1, lo + 1, n_cols))
        else:
            lo = int(np.clip(lo, 0, n_cols - 2))
            hi = int(np.clip(hi, lo + 1, n_cols))
        return lo, hi

    def _fit_row(self, row_idx: int) -> dict:
        n_cols = self.trace.shape[1]
        degree = self._degree_var.get()
        x = np.arange(n_cols)
        y = self.trace[row_idx]
        combined = self._combined_mask()

        regions = self._get_bkg_regions(row_idx)
        bnd_start = min(s for s, e in regions) if regions else 0
        bnd_stop  = max(e for s, e in regions) if regions else n_cols
        empty = {"cheb": None, "model": None, "residual": None,
                 "outlier_mask": None, "start_idx": bnd_start, "stop_idx": bnd_stop,
                 "regions": regions}

        if not regions:
            return empty

        x_fit = np.concatenate([x[s:e] for s, e in regions])
        y_fit = np.concatenate([y[s:e] for s, e in regions])
        region_mask = np.concatenate([~combined[row_idx, s:e] for s, e in regions])

        if region_mask.sum() <= degree:
            return empty

        x_valid = x_fit[region_mask]
        y_valid = y_fit[region_mask]

        try:
            if self._sigma_clip_var.get() and self._max_iters_var.get() > 0:
                cheb, outlier_local = _fit_with_outliers(
                    x_valid, y_valid, degree,
                    sigma_upper=self._sigma_upper_var.get(),
                    sigma_lower=self._sigma_lower_var.get(),
                    max_iters=self._max_iters_var.get(),
                )
            else:
                cheb = Chebyshev.fit(x_valid, y_valid, degree,
                                     domain=[float(x_valid[0]), float(x_valid[-1])])
                outlier_local = np.zeros(len(x_valid), dtype=bool)
        except Exception:
            return empty

        y_model = cheb(x)
        residual = y - y_model

        outlier_full = np.zeros(n_cols, dtype=bool)
        outlier_full[x_valid[outlier_local]] = True

        return {
            "cheb": cheb,
            "model": y_model,
            "residual": residual,
            "outlier_mask": outlier_full,
            "start_idx": bnd_start,
            "stop_idx": bnd_stop,
            "regions": regions,
        }

    def _fit_current_row(self, update_image: bool = True):
        if self.trace is None:
            return
        result = self._fit_row(self.current_row)
        self.results[self.current_row] = result
        self._draw_row_fit(self.current_row, result)
        if update_image:
            self._update_row_line()

    def _fit_profile_current_row(self):
        """Fit the Gauss-Hermite profile for the current row and update the plot."""
        if self.trace is None or not HAS_SCIPY:
            return
        if self.flux_1d is None or len(self.flux_1d) != self.trace.shape[0]:
            self.flux_1d = np.full(self.trace.shape[0], np.nan)
            self.err_1d = np.full(self.trace.shape[0], np.nan)
        i = self.current_row
        # Invalidate cache so the explicit fit always reruns
        if self.profile_fit_cache is not None and i < len(self.profile_fit_cache):
            self.profile_fit_cache[i] = None
        r = self._extraction_process_row(i)
        self._draw_extraction_row(i, r["xw"], r["yw"], r["good_pix"],
                                   r["profile_w"], r.get("params"), r["outlier_w"],
                                   r["start_idx"], r["stop_idx"],
                                   r.get("yw_interp"), r.get("rejected_in_win"))

    # ------------------------------------------------------------------
    # Unified mode dispatch
    # ------------------------------------------------------------------

    def _fit_mode_current_row(self, update_image: bool = True):
        if self._fit_mode_var is None:
            return
        if self._fit_mode_var.get() == "background":
            self._fit_current_row(update_image=update_image)
        else:
            self._fit_profile_current_row()

    def _fit_mode_start_all(self):
        if self._fit_mode_var is None:
            return
        if self._fit_mode_var.get() == "background":
            self._start_fit_all()
        else:
            self._start_extraction()

    def _fit_mode_stop(self):
        if self._fit_running:
            self._stop_fit_all()
        elif self._extr_running:
            self._stop_extraction()

    def _fit_mode_resume(self):
        if self._any_loop_running():
            return
        if self._fit_mode_var is None:
            return
        if self._fit_mode_var.get() == "background":
            if self.trace is None:
                return
            if self._fit_row_idx == 0 and all(r is None for r in self.results):
                self._fit_mode_start_all()
                return
            self._fit_stop = False
            self._fit_running = True
            self._stop_btn.config(state=tk.NORMAL)
            self._set_mode_radios_state(tk.DISABLED)
            self._fit_all_step()
        else:
            if self.trace is None or self.flux_1d is None:
                return
            if self._extr_row_idx == 0 and np.all(np.isnan(self.flux_1d)):
                self._fit_mode_start_all()
                return
            self._extr_running = True
            self._stop_btn.config(state=tk.NORMAL)
            self._set_mode_radios_state(tk.DISABLED)
            self._extraction_step()

    def _start_fit_all(self):
        if self.trace is None:
            messagebox.showwarning("No data", "Load a FITS file first.")
            return
        if self._fit_running:
            return
        if self._mode_var.get() == "trace" and self.centers is None:
            messagebox.showerror(
                "No trace defined",
                "Switch to the Fit Parameters tab and apply a trace before fitting in trace mode.",
            )
            return
        self._set_mode_radios_state(tk.DISABLED)
        self._fit_running = True
        self._fit_stop = False
        self._fit_row_idx = 0
        self._stop_btn.config(state=tk.NORMAL)
        self._progress_var.set(0)
        self._fit_all_step()

    def _fit_all_step(self):
        if self._fit_stop or self._fit_row_idx >= len(self.results):
            self._fit_running = False
            self._stop_btn.config(state=tk.DISABLED)
            self._set_mode_radios_state(tk.NORMAL)
            self._progress_var.set(100)
            self._status_var.set(
                f"Fitting complete — {len(self.results)} rows."
                if not self._fit_stop else "Fitting stopped."
            )
            return
        i = self._fit_row_idx
        result = self._fit_row(i)
        self.results[i] = result
        # Track the currently-fitted row so both plots update live
        self.current_row = i
        self._row_var.set(i)
        self._update_row_line()
        if self._live_display_var.get():
            self._draw_row_fit(i, result)
        pct = 100.0 * (i + 1) / len(self.results)
        self._progress_var.set(pct)
        self._status_var.set(f"Fitting row {i + 1} / {len(self.results)} …")
        self._fit_row_idx += 1
        self._fit_after_id = self.after(1, self._fit_all_step)

    def _stop_fit_all(self):
        self._fit_stop = True
        if self._fit_after_id is not None:
            self.after_cancel(self._fit_after_id)
            self._fit_after_id = None
        if self._param_change_after_id is not None:
            self.after_cancel(self._param_change_after_id)
            self._param_change_after_id = None
        self._fit_running = False
        self._stop_btn.config(state=tk.DISABLED)
        self._set_mode_radios_state(tk.NORMAL)
        self._status_var.set("Fitting stopped.")

    def _set_mode_radios_state(self, state):
        """Enable or disable the fit-mode radiobuttons in the nav tab."""
        for rb in self._mode_radios:
            rb.config(state=state)
        # Also keep the Fit Parameters tab full/trace radiobuttons in sync for background
        if state == tk.DISABLED:
            self._mode_full_rb.config(state=tk.DISABLED)
            self._mode_trace_rb.config(state=tk.DISABLED)
        else:
            self._mode_full_rb.config(state=tk.NORMAL)
            self._mode_trace_rb.config(state=tk.NORMAL)
        self._region_click_btn.config(state=state)

    def _any_loop_running(self) -> bool:
        return self._fit_running or self._extr_running

    # ------------------------------------------------------------------
    # Trace overlay
    # ------------------------------------------------------------------

    def _apply_trace_coeffs(self):
        if self.trace is None:
            messagebox.showwarning("No data", "Load a FITS file first.")
            return
        raw = self._trace_coeffs_var.get().strip()
        if not raw:
            messagebox.showerror("Empty input", "Enter polynomial coefficients (e.g. '0.05, 512').")
            return
        try:
            coeffs = [float(v.strip()) for v in raw.split(",")]
        except ValueError:
            messagebox.showerror("Parse error", "Coefficients must be comma-separated numbers.")
            return
        n_rows = self.trace.shape[0]
        self.centers = np.polyval(coeffs, np.arange(n_rows))
        self._trace_status_var.set(
            f"{n_rows} centers defined  "
            f"(col {self.centers.min():.1f} – {self.centers.max():.1f})"
        )
        self._draw_trace_overlay()

    def _draw_trace_overlay(self, redraw: bool = True):
        if self._trace_line is not None:
            try:
                self._trace_line.remove()
            except Exception:
                pass
            self._trace_line = None
        if self.centers is not None and self._img_im is not None:
            rows = np.arange(len(self.centers))
            (self._trace_line,) = self._img_ax.plot(
                self.centers, rows, color="cyan", lw=1.2, alpha=0.85, zorder=4,
            )
        if redraw:
            self._img_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Extraction aperture lines on image
    # ------------------------------------------------------------------

    def _draw_extr_region_lines(self, redraw: bool = True):
        """Draw yellow dashed vertical lines showing fit regions or extraction aperture."""
        for line in self._extr_vlines:
            try:
                line.remove()
            except Exception:
                pass
        self._extr_vlines = []

        if self.trace is None or self._img_im is None:
            return
        if not hasattr(self, "_extr_region_var"):
            return

        n_cols = self.trace.shape[1]
        is_profile = (self._fit_mode_var is not None and
                      self._fit_mode_var.get() == "profile")

        if is_profile:
            if self._extr_region_var.get() == "custom":
                try:
                    lo = int(self._extr_lo_var.get())
                    hi = int(self._extr_hi_var.get())
                except (tk.TclError, ValueError):
                    return
            else:
                lo, hi = self._get_fit_region(self.current_row)
            lo = max(0, min(lo, n_cols - 1))
            hi = max(lo + 1, min(hi, n_cols))
            l1 = self._img_ax.axvline(lo, color="yellow", lw=1.2, ls="--", alpha=0.85, zorder=5)
            l2 = self._img_ax.axvline(hi, color="yellow", lw=1.2, ls="--", alpha=0.85, zorder=5)
            self._extr_vlines = [l1, l2]
        else:
            for lo, hi in self._fit_regions:
                lo = max(0, min(lo, n_cols - 1))
                hi = max(lo + 1, min(hi, n_cols))
                l1 = self._img_ax.axvline(lo, color="yellow", lw=1.2, ls="--", alpha=0.85, zorder=5)
                l2 = self._img_ax.axvline(hi, color="yellow", lw=1.2, ls="--", alpha=0.85, zorder=5)
                self._extr_vlines.extend([l1, l2])

        if redraw:
            self._img_canvas.draw_idle()

    def _start_region_click(self):
        """Activate two-click mode to select the fit region on the image.

        In background mode updates Start/Stop col; in profile mode updates the
        extraction aperture lo/hi and switches to custom bounds.
        """
        if self.trace is None:
            messagebox.showwarning("No data", "Load a FITS file first.")
            return
        self._region_click_state = 1
        self._region_click_btn.config(text="Click lower bound on image…")
        self._region_click_status_var.set("Click lower (left) column boundary.")
        self._status_var.set("Select region: click the lower column boundary on the image.")

    # ------------------------------------------------------------------
    # Parameter-change guard during fitting
    # ------------------------------------------------------------------

    def _on_param_change(self, *_):
        """Fires on every trace_add write to a fitting-parameter variable."""
        if not self._fit_running:
            return
        # Immediately halt the fitting loop so no more rows run with old params
        if self._fit_after_id is not None:
            self.after_cancel(self._fit_after_id)
            self._fit_after_id = None
        self._fit_stop = True
        self._fit_running = False
        # Cancel any earlier debounce (user may still be typing)
        if self._param_change_after_id is not None:
            self.after_cancel(self._param_change_after_id)
        self._param_change_after_id = self.after(600, self._handle_param_change_while_fitting)

    def _handle_param_change_while_fitting(self):
        self._param_change_after_id = None
        self._stop_btn.config(state=tk.DISABLED)
        self._set_mode_radios_state(tk.NORMAL)
        resume_row = self._fit_row_idx
        if messagebox.askyesno(
            "Parameters changed",
            f"Fitting parameters changed while running.\n\n"
            f"Resume from row {resume_row} with the new parameters?",
        ):
            self._fit_stop = False
            self._fit_running = True
            self._set_mode_radios_state(tk.DISABLED)
            self._stop_btn.config(state=tk.NORMAL)
            self._fit_all_step()
        else:
            self._status_var.set(
                f"Fitting paused at row {resume_row}. Adjust parameters and click Fit All to restart."
            )

    # ------------------------------------------------------------------
    # Row fit plot
    # ------------------------------------------------------------------

    def _redraw_current_row(self):
        result = self.results[self.current_row] if self.results else None
        self._draw_row_fit(self.current_row, result)

    def _draw_row_fit(self, row_idx: int, result: Optional[dict]):
        if self.trace is None:
            return
        n_cols = self.trace.shape[1]
        x = np.arange(n_cols)
        y = self.trace[row_idx]
        combined = self._combined_mask()
        bad = combined[row_idx]

        ax1, ax2 = self._fit_ax1, self._fit_ax2
        ax1.clear()
        ax2.clear()

        has_fit = result is not None and result.get("model") is not None

        if has_fit:
            model = result["model"]
            residual = result["residual"]
            outlier = result.get("outlier_mask")
            regions = result.get("regions") or [(result["start_idx"], result["stop_idx"])]

            all_s = min(s for s, e in regions)
            all_e = max(e for s, e in regions)
            pad = max(2, (all_e - all_s) // 20)

            # Full span: data and residual as one continuous line
            ax1.plot(x[all_s:all_e], y[all_s:all_e], color="0.3", lw=0.8, label="Data")
            ax2.plot(x[all_s:all_e], residual[all_s:all_e], color="C3", lw=0.8)
            ax2.axhline(0, color="0.5", lw=0.8, ls="--")

            # Model across full span
            ax1.plot(x[all_s:all_e], model[all_s:all_e], color="C1", lw=1.3, label="Model")

            # Highlight selected regions
            for s, e in regions:
                ax1.axvspan(s, e, color="C0", alpha=0.18, label="_nolegend_")
                ax2.axvspan(s, e, color="C0", alpha=0.18, label="_nolegend_")

            # Bad pixel markers across full span
            bads_span = bad[all_s:all_e]
            if bads_span.any():
                ax1.scatter(x[all_s:all_e][bads_span], y[all_s:all_e][bads_span],
                            marker="x", color="purple", s=50, zorder=5,
                            linewidths=1.5, label="Bad pixel")

            # Sigma-clipped outlier markers
            if outlier is not None and outlier[all_s:all_e].any():
                ol_span = outlier[all_s:all_e]
                ax1.scatter(x[all_s:all_e][ol_span], y[all_s:all_e][ol_span],
                            marker="o", edgecolors="red", facecolors="none",
                            s=70, zorder=6, linewidths=1.5, label="Clipped")
                ax2.scatter(x[all_s:all_e][ol_span], residual[all_s:all_e][ol_span],
                            marker="o", edgecolors="red", facecolors="none",
                            s=70, zorder=6, linewidths=1.5)

            ax1.set_xlim(all_s - pad, all_e + pad)
            ax2.set_xlim(all_s - pad, all_e + pad)

            n_clipped = int(outlier[all_s:all_e].sum()) if outlier is not None else 0
            n_bad = int(bad[all_s:all_e].sum())
            ax1.set_title(
                f"Row {row_idx}  |  clipped: {n_clipped}  bad pixels: {n_bad}",
                fontsize=9,
            )
        else:
            # No fit yet — show full row
            ax1.plot(x, y, color="0.3", lw=0.8, label="Data")
            if bad.any():
                ax1.scatter(x[bad], y[bad], marker="x", color="purple",
                            s=50, zorder=5, linewidths=1.5, label="Bad pixel")
            msg = "No fit yet" if result is None else "Fit region too small / insufficient good pixels"
            ax1.set_title(f"Row {row_idx}  —  {msg}", fontsize=9)

        ax1.legend(fontsize=7, loc="upper right", framealpha=0.6)
        ax1.set_ylabel("Flux", fontsize=8)
        ax1.tick_params(labelsize=7)
        ax2.set_xlabel("Column", fontsize=8)
        ax2.set_ylabel("Residual", fontsize=8)
        ax2.tick_params(labelsize=7)

        self._fit_fig.tight_layout(pad=1.2)
        self._fit_canvas.draw_idle()

    def _draw_extraction_row(self, row_idx: int, xw: np.ndarray, yw: np.ndarray,
                              good_pix: np.ndarray, profile_w, params, outlier_w,
                              start_idx: int, stop_idx: int,
                              yw_interp: Optional[np.ndarray] = None,
                              rejected_in_win: Optional[np.ndarray] = None):
        """Show the Gauss-Hermite profile fit on the top panel during extraction."""
        ax1, ax2 = self._fit_ax1, self._fit_ax2
        ax1.clear()
        ax2.clear()

        combined = self._combined_mask()
        bad_in_ap = combined[row_idx, start_idx:stop_idx]

        # Separate masked vs clipped for distinct symbols
        clipped_in_win = np.zeros(len(xw), dtype=bool)
        if outlier_w is not None and outlier_w.any():
            good_idx = np.where(good_pix)[0]
            clipped_in_win[good_idx[outlier_w]] = True

        ax1.plot(xw, yw, color="0.3", lw=0.8, label="Data")
        ax1.axvspan(start_idx, stop_idx, color="C0", alpha=0.12, label="Aperture")

        # Masked pixels (manual + CR): purple x on original value
        if bad_in_ap.any():
            n_bad = int(bad_in_ap.sum())
            ax1.scatter(xw[bad_in_ap], yw[bad_in_ap],
                        marker="x", color="purple", s=50, zorder=5, linewidths=1.5,
                        label=f"Masked ({n_bad})")

        # Sigma-clipped pixels: open red circle on original value
        if clipped_in_win.any():
            n_clip = int(clipped_in_win.sum())
            ax1.scatter(xw[clipped_in_win], yw[clipped_in_win],
                        marker="o", edgecolors="red", facecolors="none",
                        s=60, zorder=5, linewidths=1.5,
                        label=f"Clipped ({n_clip})")

        # Interpolated replacements: small filled green diamond
        if yw_interp is not None and rejected_in_win is not None and rejected_in_win.any():
            ax1.scatter(xw[rejected_in_win], yw_interp[rejected_in_win],
                        marker="D", color="limegreen", s=25, zorder=6,
                        label="Interpolated")

        flux_val = (self.flux_1d[row_idx]
                    if self.flux_1d is not None else np.nan)
        has_fit = profile_w is not None and np.isfinite(flux_val)

        if has_fit:
            # Evaluate the raw GH profile over the full aperture using fitted params
            gh_order = self._gh_order_var.get()
            n_extra = max(0, gh_order - 2)
            h_vals = [params.get(k, 0.0) for k in ("h3", "h4")[:n_extra]]
            model_ap = _gauss_hermite_profile(
                xw.astype(float), params["amplitude"], params["mu"], params["sigma"], *h_vals)

            # clip_g: which positions within good_pix are sigma-clipped
            clip_g = clipped_in_win[good_pix] if outlier_w is None else outlier_w
            clean_g = ~clip_g  # truly clean: not masked, not clipped

            xg = xw[good_pix]
            ax1.plot(xw, model_ap, color="C1", lw=1.3, label="Profile fit")

            # Residual line — clean (non-clipped) pixels vs GH model
            model_clean = model_ap[good_pix][clean_g]
            residual_clean = yw[good_pix][clean_g] - model_clean
            ax2.plot(xg[clean_g], residual_clean, color="C3", lw=0.8)
            ax2.axhline(0, color="0.5", lw=0.8, ls="--")

            # Clipped pixels in residual panel: interpolated value vs GH model
            if clip_g.any():
                model_clip = model_ap[good_pix][clip_g]
                if yw_interp is not None and clipped_in_win.any():
                    resid_clip = yw_interp[clipped_in_win] - model_clip
                else:
                    resid_clip = yw[good_pix][clip_g] - model_clip
                ax2.scatter(xg[clip_g], resid_clip,
                            marker="o", edgecolors="red", facecolors="none",
                            s=60, zorder=6, linewidths=1.5)

            err_val = (float(self.err_1d[row_idx])
                       if self.err_1d is not None and np.isfinite(self.err_1d[row_idx])
                       else None)
            if err_val is not None:
                ax2.axhspan(-err_val, err_val, color="C0", alpha=0.15, label=f"±err ({err_val:.3g})")
                ax2.legend(fontsize=7, loc="upper right", framealpha=0.6)

            n_clip = int(clipped_in_win.sum())
            ax1.set_title(
                f"Row {row_idx}  [extract]  |  flux: {flux_val:.4g}"
                + (f"  masked: {int(bad_in_ap.sum())}" if bad_in_ap.any() else "")
                + (f"  clipped: {n_clip}" if n_clip else ""),
                fontsize=9,
            )
        else:
            ax1.set_title(f"Row {row_idx}  [extract]  —  profile fit failed", fontsize=9)

        ax1.legend(fontsize=7, loc="upper right", framealpha=0.6)
        ax1.set_ylabel("Flux", fontsize=8)
        ax1.tick_params(labelsize=7)
        ax2.set_xlabel("Column", fontsize=8)
        ax2.set_ylabel("Residual", fontsize=8)
        ax2.tick_params(labelsize=7)

        self._fit_fig.tight_layout(pad=1.2)
        self._fit_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Bad pixel handling
    # ------------------------------------------------------------------

    def _toggle_row_flag(self):
        if self.trace is None or self.row_flag is None:
            return
        i = self.current_row
        self.row_flag[i] = not self.row_flag[i]
        self._update_flag_btn()
        self._update_flag_overlay()

    def _update_flag_btn(self):
        if self.row_flag is None:
            self._flag_btn_var.set("⚑ Flag Row")
            self._flag_status_var.set("")
            return
        flagged = self.row_flag[self.current_row]
        self._flag_btn_var.set("✖ Unflag Row" if flagged else "⚑ Flag Row")
        n_total = int(self.row_flag.sum())
        self._flag_status_var.set(f"[FLAGGED]  {n_total} flagged" if flagged
                                   else (f"{n_total} flagged" if n_total else ""))

    def _update_flag_overlay(self):
        if self._img_ax is None:
            return
        if self._flag_scatter is not None:
            try:
                self._flag_scatter.remove()
            except Exception:
                pass
            self._flag_scatter = None
        if self.row_flag is not None and self.row_flag.any():
            flagged_rows = np.where(self.row_flag)[0]
            # Draw a short bar at the left edge of each flagged row
            self._flag_scatter = self._img_ax.scatter(
                np.zeros(len(flagged_rows)),
                flagged_rows,
                c="orange", s=12, marker=">", zorder=4, linewidths=0,
            )
        self._img_canvas.draw_idle()

    def _toggle_bad_pixel_mode(self):
        self.bad_pixel_mode = not self.bad_pixel_mode
        state = "ON" if self.bad_pixel_mode else "OFF"
        self._bp_btn_text.set(f"Bad Pixel Mode: {state}")
        self._bp_status_var.set(f"Bad pixel mode: {state}")
        color = "red" if self.bad_pixel_mode else "gray"
        self._status_var.set(
            f"Bad pixel mode {state}. Click the image or row plot to toggle pixels."
            if self.bad_pixel_mode
            else "Bad pixel mode OFF. Click image to navigate rows."
        )

    def _on_image_click(self, event):
        if event.inaxes != self._img_ax or self.trace is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        n_rows, n_cols = self.trace.shape
        if not (0 <= col < n_cols and 0 <= row < n_rows):
            return

        # Right-click: remove nearest background region
        if event.button == 3:
            is_profile = (self._fit_mode_var is not None and
                          self._fit_mode_var.get() == "profile")
            if not is_profile:
                self._remove_region_at_col(col)
            return

        # Shared region-select click mode (background: adds region; profile: updates aperture)
        if self._region_click_state in (1, 2):
            is_profile = (self._fit_mode_var is not None and
                          self._fit_mode_var.get() == "profile")
            if self._region_click_state == 1:
                if is_profile:
                    self._extr_lo_var.set(col)
                    self._extr_region_var.set("custom")
                else:
                    self._region_click_lo = col
                self._region_click_state = 2
                self._region_click_btn.config(text="Click upper bound on image…")
                self._region_click_status_var.set("Click upper (right) column boundary.")
                self._status_var.set("Select region: now click the upper column boundary.")
            else:
                if is_profile:
                    lo = self._extr_lo_var.get()
                    hi = col
                    if hi < lo:
                        lo, hi = hi, lo
                    self._extr_lo_var.set(lo)
                    self._extr_hi_var.set(hi)
                    self._extr_region_var.set("custom")
                    label = f"Aperture set: col {lo} – {hi}"
                    msg = f"Extraction aperture set to columns {lo} – {hi}."
                else:
                    lo = self._region_click_lo if self._region_click_lo is not None else 0
                    hi = col
                    if hi < lo:
                        lo, hi = hi, lo
                    self._fit_regions.append((lo, hi))
                    self._fit_regions.sort()
                    self._refresh_regions_listbox()
                    label = f"Added region: col {lo} – {hi}"
                    msg = f"Region col {lo} – {hi} added. Click 'Select region' again for another."
                self._region_click_lo = None
                self._region_click_state = 0
                self._region_click_btn.config(text="Select region on image")
                self._region_click_status_var.set(label)
                self._status_var.set(msg)
                self._draw_extr_region_lines()
                if not is_profile and self._autofit_var.get():
                    self._fit_mode_current_row(update_image=False)
            return

        if self.bad_pixel_mode:
            self._toggle_pixel_mask(row, col)
            self._redraw_mask_overlay()
            if row == self.current_row:
                self._fit_mode_current_row(update_image=False)
        else:
            self._go_to_row(row)

    def _on_fit_click(self, event):
        if event.inaxes not in (self._fit_ax1, self._fit_ax2):
            return
        if not self.bad_pixel_mode or self.trace is None:
            return
        col = int(round(event.xdata))
        n_cols = self.trace.shape[1]
        if not (0 <= col < n_cols):
            return
        self._toggle_pixel_mask(self.current_row, col)
        self._redraw_mask_overlay()
        self._fit_mode_current_row(update_image=False)

    # ------------------------------------------------------------------
    # Mouse-position tracking and 'd' key handlers
    # ------------------------------------------------------------------
    def _on_image_motion(self, event):
        """Track cursor position over the image canvas."""
        self._img_cursor_row = int(round(event.ydata)) if event.ydata is not None else None
        self._img_cursor_col = int(round(event.xdata)) if event.xdata is not None else None

    def _on_fit_motion(self, event):
        """Track cursor position over the fit canvas."""
        self._fit_cursor_col = int(round(event.xdata)) if event.xdata is not None else None

    def _on_image_key(self, event):
        """Handle key presses while cursor is over the image canvas."""
        if event.key != "d" or not self.bad_pixel_mode or self.trace is None:
            return
        row = getattr(self, "_img_cursor_row", None)
        col = getattr(self, "_img_cursor_col", None)
        if row is None or col is None:
            return
        n_rows, n_cols = self.trace.shape
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            return
        self._toggle_pixel_mask(row, col)
        self._redraw_mask_overlay()
        if row == self.current_row:
            self._fit_mode_current_row(update_image=False)

    def _on_fit_key(self, event):
        """Handle key presses while cursor is over the fit canvas."""
        if event.key != "d" or not self.bad_pixel_mode or self.trace is None:
            return
        col = getattr(self, "_fit_cursor_col", None)
        if col is None:
            return
        n_cols = self.trace.shape[1]
        if not (0 <= col < n_cols):
            return
        self._toggle_pixel_mask(self.current_row, col)
        self._redraw_mask_overlay()
        self._fit_mode_current_row(update_image=False)

    def _toggle_pixel_mask(self, row: int, col: int):
        """Toggle bad/good for a single pixel.

        If the pixel is in cr_mask, clicking cycles: masked → unmasked → masked.
        If it's not in cr_mask, clicking toggles manual_mask.
        """
        if self.cr_mask is not None and self.cr_mask[row, col]:
            if self.cr_unmask is None:
                self.cr_unmask = np.zeros(self.trace.shape, dtype=bool)
            self.cr_unmask[row, col] = not self.cr_unmask[row, col]
        else:
            self.manual_mask[row, col] = not self.manual_mask[row, col]
        if self.profile_fit_cache is not None and row < len(self.profile_fit_cache):
            self.profile_fit_cache[row] = None

    def _peek_cr_press(self):
        self._cr_hidden = True
        self._redraw_mask_overlay()

    def _peek_cr_release(self):
        self._cr_hidden = False
        self._redraw_mask_overlay()

    def _clear_row_mask(self):
        if self.manual_mask is None or self.trace is None:
            return
        i = self.current_row
        self.manual_mask[i, :] = False
        if self.profile_fit_cache is not None and i < len(self.profile_fit_cache):
            self.profile_fit_cache[i] = None
        self._redraw_mask_overlay()
        self._fit_mode_current_row(update_image=False)

    def _clear_all_masks(self):
        if self.manual_mask is None or self.trace is None:
            return
        if messagebox.askyesno("Confirm", "Clear all manually marked bad pixels?"):
            self.manual_mask[:] = False
            if self.profile_fit_cache is not None:
                self.profile_fit_cache = [None] * len(self.profile_fit_cache)
            self._redraw_mask_overlay()
            self._redraw_current_row()


# ---------------------------------------------------------------------------
# Optimal extraction — new methods added below _clear_all_masks
# ---------------------------------------------------------------------------

    def _build_extract_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        r = 0

        # --- Variance input ---
        var_lf = ttk.LabelFrame(parent, text="Variance Input", padding=4)
        var_lf.grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        var_lf.columnconfigure(1, weight=1)
        r += 1

        var_type_row = ttk.Frame(var_lf)
        var_type_row.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self._var_type_var = tk.StringVar(value="ivar")
        ttk.Radiobutton(var_type_row, text="IVAR", variable=self._var_type_var, value="ivar").pack(side=tk.LEFT)
        ttk.Radiobutton(var_type_row, text="ERR", variable=self._var_type_var, value="err").pack(side=tk.LEFT, padx=(8, 0))
        self._make_help_btn(var_type_row,
            "Select the type of uncertainty array stored in the FITS extension:\n\n"
            "  IVAR  — inverse variance (1/σ²). Optimal extraction weights pixels by IVAR.\n"
            "  ERR   — standard error (σ). Will be squared internally to produce IVAR.\n\n"
            "If no variance is loaded, all pixels are given equal weight.").pack(side=tk.LEFT, padx=(6, 0))

        ttk.Button(var_lf, text="Load from current file",
                   command=self._load_variance_current).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        ttk.Button(var_lf, text="Load from other file…",
                   command=self._load_variance_other).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        self._var_file_var = tk.StringVar(value="")
        ttk.Label(var_lf, textvariable=self._var_file_var,
                  font=("TkDefaultFont", 8), foreground="gray").grid(
            row=3, column=0, columnspan=2, sticky=tk.W)

        self._var_status_var = tk.StringVar(value="No variance loaded — uniform weights used.")
        ttk.Label(var_lf, textvariable=self._var_status_var,
                  font=("TkDefaultFont", 8), foreground="gray").grid(
            row=4, column=0, columnspan=2, sticky=tk.W)

        # --- Profile fitting ---
        prof_lf = ttk.LabelFrame(parent, text="Gauss-Hermite Profile", padding=4)
        prof_lf.grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        prof_lf.columnconfigure(1, weight=1)
        r += 1

        gh_row = ttk.Frame(prof_lf)
        gh_row.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(gh_row, text="GH order N (2=Gaussian):").pack(side=tk.LEFT)
        self._gh_order_var = tk.IntVar(value=4)
        ttk.Spinbox(gh_row, from_=2, to=10, textvariable=self._gh_order_var, width=4).pack(
            side=tk.LEFT, padx=4)
        self._make_help_btn(gh_row,
            "Gauss-Hermite order N controls how many moments are fit:\n\n"
            "  N=2  Gaussian only (amplitude, centre, width)\n"
            "  N=3  +h3 (skewness)\n"
            "  N=4  +h3, h4 (kurtosis)\n\n"
            "Sigma clipping settings are in the Common tab.").pack(side=tk.LEFT)

        fill_row = ttk.Frame(prof_lf)
        fill_row.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        ttk.Label(fill_row, text="At rejected pixels use:").pack(side=tk.LEFT)
        self._rejection_fill_var = tk.StringVar(value="spline")
        ttk.Radiobutton(fill_row, text="Spline interp",
                        variable=self._rejection_fill_var, value="spline").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(fill_row, text="GH model",
                        variable=self._rejection_fill_var, value="model").pack(side=tk.LEFT, padx=(6, 0))
        self._make_help_btn(fill_row,
            "How to fill rejected (sigma-clipped) pixels before the optimal-extraction sum:\n\n"
            "  Spline interp — interpolate from neighbouring good pixels along the column.\n"
            "  GH model — substitute the best-fit Gauss-Hermite profile value.\n\n"
            "Spline is usually safer when bad pixels cluster at the spectrum edge; "
            "GH model is better when the profile is well-constrained.").pack(side=tk.LEFT, padx=(4, 0))

        # --- Extraction region ---
        reg_lf = ttk.LabelFrame(parent, text="Extraction Aperture", padding=4)
        reg_lf.grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        reg_lf.columnconfigure(1, weight=1)
        r += 1

        self._extr_region_var = tk.StringVar(value="fit")
        ttk.Radiobutton(reg_lf, text="Use background-fit region",
                        variable=self._extr_region_var, value="fit",
                        command=self._draw_extr_region_lines).grid(
            row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Radiobutton(reg_lf, text="Custom bounds",
                        variable=self._extr_region_var, value="custom",
                        command=self._draw_extr_region_lines).grid(
            row=1, column=0, columnspan=2, sticky=tk.W)

        ttk.Label(reg_lf, text="Col low:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self._extr_lo_var = tk.IntVar(value=0)
        ttk.Entry(reg_lf, textvariable=self._extr_lo_var, width=7).grid(
            row=2, column=1, sticky=tk.W, padx=4)

        ttk.Label(reg_lf, text="Col high:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self._extr_hi_var = tk.IntVar(value=100)
        ttk.Entry(reg_lf, textvariable=self._extr_hi_var, width=7).grid(
            row=3, column=1, sticky=tk.W, padx=4)

        for v in (self._extr_lo_var, self._extr_hi_var):
            v.trace_add("write", lambda *_: self._draw_extr_region_lines())

        aper_hint_row = ttk.Frame(reg_lf)
        aper_hint_row.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        self._make_help_btn(aper_hint_row,
            "Yellow dashed lines on the image show the extraction aperture.\n\n"
            "In Trace mode the aperture window follows the traced centre column per row.\n\n"
            "Use 'Select region on image' in the Common tab to set bounds by clicking.").pack(side=tk.LEFT)

        # --- Save ---
        save_lf = ttk.LabelFrame(parent, text="Save", padding=4)
        save_lf.grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        r += 1

        ttk.Button(save_lf, text="Save 1D FITS",
                   command=self._save_1d_fits).pack(side=tk.LEFT, padx=2)
        self._make_help_btn(save_lf,
            "Save the extracted 1D spectrum as a FITS file.\n\n"
            "Output contains two extensions: FLUX and ERR.\n"
            "Run 'Fit All Rows' in the Navigate & Fit tab first to produce results.").pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # Variance loading
    # ------------------------------------------------------------------

    def _pick_extension(self, fits_path: str) -> Optional[int]:
        """Return the HDU index to load for variance/err.

        If only one 2D image extension exists, return it silently.
        If multiple exist, show a selection dialog.
        """
        try:
            with _fits.open(fits_path) as hdul:
                image_exts = [
                    i for i, hdu in enumerate(hdul)
                    if hdu.data is not None and np.asarray(hdu.data).ndim == 2
                ]
                if len(image_exts) == 0:
                    messagebox.showerror("Error", "No 2D image data found in this FITS file.")
                    return None
                if len(image_exts) == 1:
                    return image_exts[0]
                labels = []
                for i in image_exts:
                    hdu = hdul[i]
                    name = hdu.name or f"HDU {i}"
                    shape = np.asarray(hdu.data).shape
                    labels.append(f"[{i}]  {name}  {shape[0]}×{shape[1]}")
        except Exception as e:
            messagebox.showerror("Read error", str(e))
            return None

        return _ask_extension(self, labels, image_exts)

    def _load_variance_current(self):
        if self.fits_path is None or self.trace is None:
            messagebox.showwarning("No file", "Load a FITS file first.")
            return
        ext = self._pick_extension(self.fits_path)
        if ext is None:
            return
        try:
            with _fits.open(self.fits_path) as hdul:
                data = np.asarray(hdul[ext].data, dtype=float)
        except Exception as e:
            messagebox.showerror("Read error", str(e))
            return
        self._store_variance(data, self.fits_path, ext)

    def _load_variance_other(self):
        if not HAS_FITS:
            messagebox.showerror("Missing dependency", "astropy is required.")
            return
        path = filedialog.askopenfilename(
            title="Open variance FITS file",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")],
        )
        if not path:
            return
        ext = self._pick_extension(path)
        if ext is None:
            return
        try:
            with _fits.open(path) as hdul:
                data = np.asarray(hdul[ext].data, dtype=float)
        except Exception as e:
            messagebox.showerror("Read error", str(e))
            return
        self._store_variance(data, path, ext)

    def _store_variance(self, data: np.ndarray, path: str = "", ext: int = 0):
        if self.trace is None:
            return
        if data.shape != self.trace.shape:
            messagebox.showerror(
                "Shape mismatch",
                f"Variance shape {data.shape} ≠ data shape {self.trace.shape}.",
            )
            return
        vtype = self._var_type_var.get()
        if vtype == "err":
            with np.errstate(divide="ignore", invalid="ignore"):
                ivar = np.where(data > 0, 1.0 / data ** 2, 0.0)
        else:  # ivar
            ivar = np.where(data > 0, data, 0.0)
        self.ivar = ivar
        n_bad = int((ivar == 0).sum())
        self._var_file_var.set(
            f"File: {os.path.basename(path)}  ext [{ext}]"
        )
        self._var_status_var.set(
            f"{'IVAR' if vtype == 'ivar' else 'ERR'} loaded  "
            f"({n_bad} zero-weight pixels)"
        )

    # ------------------------------------------------------------------
    # Profile fitting and optimal extraction
    # ------------------------------------------------------------------

    def _fit_profile_row(self, x: np.ndarray, y: np.ndarray,
                         ivar_row: Optional[np.ndarray],
                         p0_prev: Optional[list] = None) -> tuple:
        """
        Fit a Gauss-Hermite profile to a spatial row with sigma clipping.

        Returns (profile_norm, params, outlier_mask) where profile_norm sums to 1,
        or (None, None, None) on failure.
        """
        if not HAS_SCIPY:
            return None, None, None

        gh_order = self._gh_order_var.get()
        n_extra = max(0, gh_order - 2)   # number of h terms: h3, h4, ..., h_{gh_order}
        do_clip = self._sigma_clip_var.get()
        n_sigma = self._sigma_upper_var.get()
        max_iters = self._max_iters_var.get()

        # Initial guess from weighted moments
        ypos = np.where(y > 0, y, 0.0)
        total = ypos.sum()
        if total <= 0:
            return None, None, None
        mu0 = float((x * ypos).sum() / total)
        sigma0 = max(2.0, float(np.sqrt(((x - mu0) ** 2 * ypos).sum() / total)))
        amp0 = float(y.max())

        # Build profile function dynamically for any GH order
        def prof_fn(x_, a, mu, sig, *h_extra):
            return _gauss_hermite_profile(x_, a, mu, sig, *h_extra)

        p0 = [amp0, mu0, sigma0] + [0.0] * n_extra
        lo = [0.0, x[0] - 10, 0.3] + [-0.5] * n_extra
        hi = [np.inf, x[-1] + 10, 60.0] + [0.5] * n_extra

        good = np.ones(len(y), dtype=bool)
        # Use warm-start if provided and length matches; fall back to moment-based p0
        if p0_prev is not None and len(p0_prev) == len(p0):
            popt = list(p0_prev)
        else:
            popt = list(p0)
        outlier = np.zeros(len(y), dtype=bool)

        for _ in range(max(1, max_iters)):
            if good.sum() < len(p0) + 1:
                break
            # Uncertainty weights for curve_fit
            if ivar_row is not None:
                w = ivar_row[good]
                sigma_w = np.where(w > 0, 1.0 / np.sqrt(w), 1e10)
            else:
                sigma_w = None
            try:
                popt, _ = _scipy_curve_fit(
                    prof_fn, x[good], y[good], p0=popt,
                    sigma=sigma_w, absolute_sigma=(sigma_w is not None),
                    bounds=(lo, hi), maxfev=3000,
                )
            except Exception:
                break

            if not do_clip:
                break
            resid = y - prof_fn(x, *popt)
            std = float(np.std(resid[good]))
            if std == 0.0:
                break
            good = np.abs(resid) <= n_sigma * std
            outlier = ~good

        # Evaluate and normalise profile (clip negatives before normalising)
        profile = np.maximum(prof_fn(x, *popt), 0.0)
        psum = profile.sum()
        if psum <= 0:
            return None, None, None
        profile /= psum

        param_keys = ["amplitude", "mu", "sigma", "h3", "h4"]
        params = {k: float(v) for k, v in zip(param_keys, popt)}
        return profile, params, outlier

    def _extraction_process_row(self, i: int, p0_prev: Optional[list] = None) -> dict:
        """Compute the Gauss-Hermite profile fit for row i.

        Writes self.flux_1d[i] and self.err_1d[i] as side effects.
        Returns a dict with keys: xw, yw, good_pix, profile_w, outlier_w,
        start_idx, stop_idx — for use by _draw_extraction_row.
        """
        # Return cached result if mask hasn't changed since last fit
        if (self.profile_fit_cache is not None
                and i < len(self.profile_fit_cache)
                and self.profile_fit_cache[i] is not None):
            return self.profile_fit_cache[i]

        n_cols = self.trace.shape[1]
        x = np.arange(n_cols)
        y = self.trace[i].copy()

        if self._extr_region_var.get() == "custom":
            start_idx = max(0, min(int(self._extr_lo_var.get()), n_cols - 2))
            stop_idx = max(start_idx + 1, min(int(self._extr_hi_var.get()), n_cols))
        else:
            start_idx, stop_idx = self._get_fit_region(i)

        xw = x[start_idx:stop_idx]
        yw = y[start_idx:stop_idx]
        ivar_row_w = (self.ivar[i, start_idx:stop_idx]
                      if self.ivar is not None else None)

        # Clear any previous sigma-clip results for this row before computing good_pix,
        # so a refit with clipping disabled does not exclude previously clipped pixels.
        if self.clip_mask is not None:
            self.clip_mask[i, :] = False

        combined = self._combined_mask()
        good_pix = ~combined[i, start_idx:stop_idx]

        empty = {"xw": xw, "yw": yw, "good_pix": good_pix,
                 "profile_w": None, "outlier_w": None,
                 "start_idx": start_idx, "stop_idx": stop_idx}

        if good_pix.sum() < 5:
            if self.profile_fit_cache is not None and i < len(self.profile_fit_cache):
                self.profile_fit_cache[i] = empty
            return empty

        ivar_w = ivar_row_w[good_pix] if ivar_row_w is not None else None
        profile_w, params, outlier_w = self._fit_profile_row(
            xw[good_pix], yw[good_pix], ivar_w, p0_prev=p0_prev,
        )

        # Store per-row params for post-hoc outlier correction
        if self.profile_params_rows is not None and i < len(self.profile_params_rows):
            self.profile_params_rows[i] = params

        # Record newly clipped pixels into clip_mask
        if self.clip_mask is None:
            self.clip_mask = np.zeros(self.trace.shape, dtype=bool)
        if outlier_w is not None and outlier_w.any():
            good_in_window = np.where(good_pix)[0]
            clipped_cols = good_in_window[outlier_w] + start_idx
            self.clip_mask[i, clipped_cols] = True

        # Build aperture-wide rejection mask (masked pixels + sigma-clipped pixels)
        bad_in_ap = combined[i, start_idx:stop_idx]
        clipped_in_win = np.zeros(len(xw), dtype=bool)
        if outlier_w is not None and outlier_w.any():
            good_in_window = np.where(good_pix)[0]
            clipped_in_win[good_in_window[outlier_w]] = True
        rejected_in_win = bad_in_ap | clipped_in_win

        if profile_w is not None:
            # Re-evaluate the GH profile over the full aperture and renormalize
            gh_order = self._gh_order_var.get()
            n_extra = max(0, gh_order - 2)
            h_extra_vals = [params.get(k, 0.0) for k in ("h3", "h4")[:n_extra]]
            prof_ap = np.maximum(
                _gauss_hermite_profile(xw.astype(float), params["amplitude"],
                                       params["mu"], params["sigma"], *h_extra_vals), 0.0)
            psum_ap = prof_ap.sum()
            if psum_ap > 0:
                prof_ap /= psum_ap

            profile_full = np.zeros(n_cols)
            profile_full[start_idx:stop_idx] = prof_ap

        # Fill rejected pixels for display and flux extraction
        yw_interp = yw.copy()
        clean_in_win = ~rejected_in_win
        if rejected_in_win.any():
            fill = self._rejection_fill_var.get()
            if fill == "model" and profile_w is not None:
                # Scale the GH model to a rough flux estimate from clean pixels
                flux_est = (float(np.sum(yw[clean_in_win])) /
                            float(prof_ap[clean_in_win].sum())
                            if prof_ap[clean_in_win].sum() > 0 else 1.0)
                yw_interp[rejected_in_win] = flux_est * prof_ap[rejected_in_win]
            elif clean_in_win.sum() >= 2:
                xc = xw[clean_in_win].astype(float)
                yc = yw[clean_in_win].astype(float)
                kind = "cubic" if len(xc) >= 4 else "linear"
                interp_fn = _scipy_interp1d(
                    xc, yc, kind=kind, bounds_error=False,
                    fill_value=(yc[0], yc[-1]),
                )
                yw_interp[rejected_in_win] = interp_fn(xw[rejected_in_win].astype(float))

        if profile_w is not None:
            # y_for_extraction uses filled values at rejected positions
            y_for_extraction = y.copy()
            if rejected_in_win.any():
                rej_cols = np.where(rejected_in_win)[0] + start_idx
                y_for_extraction[rej_cols] = yw_interp[rejected_in_win]

            if self.ivar is not None:
                ivar_full = np.maximum(self.ivar[i].copy(), 0.0)
            else:
                ivar_full = np.ones(n_cols)

            denom = float(np.sum(profile_full ** 2 * ivar_full))
            if denom > 0:
                self.flux_1d[i] = float(
                    np.sum(profile_full * y_for_extraction * ivar_full) / denom
                )
                if self.ivar is not None:
                    self.err_1d[i] = float(np.sqrt(1.0 / denom))
                else:
                    # No variance map: use std of residuals on clean pixels
                    model_clean = self.flux_1d[i] * profile_w
                    resid_clean = yw[good_pix] - model_clean
                    if outlier_w is not None:
                        resid_clean = resid_clean[~outlier_w]
                    self.err_1d[i] = float(np.std(resid_clean)) if len(resid_clean) > 1 else np.nan

        result = {"xw": xw, "yw": yw, "yw_interp": yw_interp,
                  "good_pix": good_pix, "rejected_in_win": rejected_in_win,
                  "profile_w": profile_w, "params": params,
                  "outlier_w": outlier_w,
                  "start_idx": start_idx, "stop_idx": stop_idx}
        if self.profile_fit_cache is not None and i < len(self.profile_fit_cache):
            self.profile_fit_cache[i] = result
        return result

    def _start_extraction(self):
        if self.trace is None:
            messagebox.showwarning("No data", "Load a FITS file first.")
            return
        if not HAS_SCIPY:
            messagebox.showerror(
                "Missing dependency",
                "scipy is required for Gauss-Hermite profile fitting.\n"
                "Install it with:  pip install scipy",
            )
            return
        if self._extr_running:
            return
        n_rows = self.trace.shape[0]
        self.flux_1d = np.full(n_rows, np.nan)
        self.err_1d = np.full(n_rows, np.nan)
        self._extr_running = True
        self._extr_row_idx = 0
        self.profile_params_rows = [None] * n_rows
        self.profile_fit_cache = [None] * n_rows
        self._stop_btn.config(state=tk.NORMAL)
        self._set_mode_radios_state(tk.DISABLED)
        self._progress_var.set(0)
        self._extraction_step()

    def _extraction_step(self):
        if self._extr_row_idx >= len(self.flux_1d):
            self._extr_running = False
            self._stop_btn.config(state=tk.DISABLED)
            self._set_mode_radios_state(tk.NORMAL)
            self._progress_var.set(100)
            self._status_var.set("Extraction complete. Click Display to view the spectrum.")
            return

        i = self._extr_row_idx
        r = self._extraction_process_row(i)
        self.current_row = i
        self._row_var.set(i)
        self._update_row_line()
        if self._live_display_var.get():
            self._draw_extraction_row(i, r["xw"], r["yw"], r["good_pix"],
                                       r["profile_w"], r.get("params"), r["outlier_w"],
                                       r["start_idx"], r["stop_idx"],
                                       r.get("yw_interp"), r.get("rejected_in_win"))
        pct = 100.0 * (i + 1) / len(self.flux_1d)
        self._progress_var.set(pct)
        self._status_var.set(f"Extracting row {i + 1} / {len(self.flux_1d)} …")
        self._extr_row_idx += 1
        self._extr_after_id = self.after(1, self._extraction_step)

    def _stop_extraction(self):
        self._extr_running = False
        if self._extr_after_id is not None:
            self.after_cancel(self._extr_after_id)
            self._extr_after_id = None
        self._stop_btn.config(state=tk.DISABLED)
        self._set_mode_radios_state(tk.NORMAL)
        self._status_var.set("Extraction stopped.")

    def _show_centroid_plot(self):
        """Popup: profile centroid (mu) vs row. Click a point to navigate to that row."""
        if self.profile_params_rows is None or self.trace is None:
            messagebox.showwarning("No fits", "Run 'Fit All Rows' in Profile mode first.")
            return

        n_rows = self.trace.shape[0]
        rows = np.arange(n_rows)
        mus = np.full(n_rows, np.nan)
        for i, p in enumerate(self.profile_params_rows):
            if p is not None:
                mus[i] = p["mu"]

        # Reuse existing window if still open
        win_alive = (self._display_win_centroid is not None and
                     self._display_win_centroid.winfo_exists())
        if win_alive:
            ax = self._display_ax_centroid
            ax.clear()
            self._draw_centroid_axes(ax, rows, mus)
            self._display_canvas_centroid.draw_idle()
            self._display_win_centroid.lift()
            return

        win = tk.Toplevel(self)
        win.title("Profile Centroid vs Row")
        win.geometry("800x380")
        self._display_win_centroid = win

        fig = Figure(figsize=(8, 3.6), dpi=90)
        ax = fig.add_subplot(111)
        self._display_ax_centroid = ax
        self._draw_centroid_axes(ax, rows, mus)
        fig.tight_layout(pad=0.8)

        canvas = FigureCanvasTkAgg(fig, master=win)
        self._display_canvas_centroid = canvas
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def _on_click(event):
            if toolbar.mode or event.inaxes != ax:
                return
            row = int(round(event.xdata))
            if 0 <= row < n_rows:
                self._go_to_row(row)
                self.lift()
                # Update the row marker
                for line in ax.lines:
                    if getattr(line, "_centroid_marker", False):
                        line.remove()
                vl = ax.axvline(row, color="red", lw=1.2, ls="--", zorder=3)
                vl._centroid_marker = True
                canvas.draw_idle()

        canvas.mpl_connect("button_press_event", _on_click)
        canvas.draw()

    def _draw_centroid_axes(self, ax, rows, mus):
        """Populate the centroid axes (used for both first draw and refresh)."""
        finite = np.isfinite(mus)
        ax.plot(rows[finite], mus[finite], color="C0", lw=0.9, marker=".", ms=3,
                label="centroid μ")
        # Highlight jumps > 1 px relative to nearest neighbour
        dmu = np.abs(np.diff(mus))
        jump_rows = np.where(dmu > 1.0)[0]   # row i where |mu[i+1]-mu[i]|>1
        if len(jump_rows):
            flagged = np.unique(np.concatenate([jump_rows, jump_rows + 1]))
            flagged = flagged[flagged < len(mus)]
            valid = flagged[np.isfinite(mus[flagged])]
            ax.scatter(rows[valid], mus[valid], color="red", s=20, zorder=4,
                       label=f"jump >1 px ({len(valid)})")
        ax.set_xlabel("Row", fontsize=8)
        ax.set_ylabel("Centroid column (px)", fontsize=8)
        ax.set_title("Profile centroid — click to navigate", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="best")

    # ------------------------------------------------------------------
    # 1D spectrum display (popup window)
    # ------------------------------------------------------------------

    def _display_stored_spectrum(self):
        """Display button: show stored 2D model (background) or 1D spectrum (profile)."""
        if self._fit_mode_var is None:
            return
        if self._fit_mode_var.get() == "background":
            self._draw_2d_spectrum()
        else:
            self._draw_1d_spectrum()

    def _draw_2d_spectrum(self):
        if not any(r is not None and r.get("model") is not None for r in self.results):
            messagebox.showinfo("No data", "No background fit results to display yet.")
            return
        n_cols = self.trace.shape[1]
        bkgsub_rows = []
        for i, r in enumerate(self.results):
            if r is None or r.get("model") is None:
                bkgsub_rows.append(np.full(n_cols, np.nan))
            else:
                regions = r.get("regions") or [(r["start_idx"], r["stop_idx"])]
                lo = min(s for s, e in regions)
                hi = max(e for s, e in regions)
                row_arr = np.zeros(n_cols)
                row_arr[lo:hi] = self.trace[i, lo:hi] - r["model"][lo:hi]
                bkgsub_rows.append(row_arr)
        bkgsub_arr = np.vstack(bkgsub_rows)
        finite_vals = bkgsub_arr[np.isfinite(bkgsub_arr)]
        vmin, vmax = _zscale(finite_vals.reshape(-1, 1)) if finite_vals.size else (0.0, 1.0)

        # Reuse existing window if still open
        win_alive = (self._display_win_2d is not None and
                     self._display_win_2d.winfo_exists())
        if win_alive:
            ax = self._display_ax_2d
            ax.clear()
            ax.imshow(bkgsub_arr, origin="lower", aspect="auto",
                      vmin=vmin, vmax=vmax, cmap="gray", interpolation="nearest")
            ax.set_xlabel("Column", fontsize=8)
            ax.set_ylabel("Row", fontsize=8)
            ax.set_title("Background-subtracted spectrum", fontsize=9)
            self._display_canvas_2d.draw_idle()
            self._display_win_2d.lift()
            return

        win = tk.Toplevel(self)
        win.title("Background-subtracted Spectrum (2D)")
        win.geometry("900x500")
        self._display_win_2d = win

        fig = Figure(figsize=(9, 5), dpi=90)
        ax = fig.add_subplot(111)
        self._display_ax_2d = ax
        ax.imshow(bkgsub_arr, origin="lower", aspect="auto",
                  vmin=vmin, vmax=vmax, cmap="gray", interpolation="nearest")
        ax.set_xlabel("Column", fontsize=8)
        ax.set_ylabel("Row", fontsize=8)
        ax.set_title("Background-subtracted spectrum", fontsize=9)
        fig.tight_layout(pad=0.8)

        canvas = FigureCanvasTkAgg(fig, master=win)
        self._display_canvas_2d = canvas
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def _on_click_2d(event):
            if toolbar.mode or event.inaxes != ax:
                return
            row = int(round(event.ydata))
            n_rows = self.trace.shape[0]
            if 0 <= row < n_rows:
                self._go_to_row(row)
                self.lift()

        canvas.mpl_connect("button_press_event", _on_click_2d)
        canvas.draw()

    def _draw_1d_spectrum(self):
        if self.flux_1d is None or not np.any(np.isfinite(self.flux_1d)):
            messagebox.showinfo("No data", "No extracted spectrum to display yet.")
            return

        rows = np.arange(len(self.flux_1d))
        finite = np.isfinite(self.flux_1d)
        err = (self.err_1d.copy() if self.err_1d is not None else None)
        if err is not None:
            err[~np.isfinite(err)] = 0.0

        # Reuse existing window if still open
        win_alive = (self._display_win_1d is not None and
                     self._display_win_1d.winfo_exists())
        if win_alive:
            ax = self._display_ax_1d
            ax.clear()
            ax.plot(rows[finite], self.flux_1d[finite], color="C0", lw=0.9, label="Flux")
            if err is not None and np.any(err > 0):
                ax.fill_between(rows[finite],
                                (self.flux_1d - err)[finite],
                                (self.flux_1d + err)[finite],
                                color="C0", alpha=0.25, label="±1σ")
            ax.set_xlabel("Row (wavelength pixel)")
            ax.set_ylabel("Flux")
            ax.legend(loc="upper right")
            ax.set_title("Extracted 1D Spectrum")
            self._display_canvas_1d.draw_idle()
            self._display_win_1d.lift()
            return

        win = tk.Toplevel(self)
        win.title("Extracted 1D Spectrum")
        win.geometry("900x420")
        self._display_win_1d = win

        fig = Figure(figsize=(9, 4), dpi=90)
        ax = fig.add_subplot(111)
        self._display_ax_1d = ax
        ax.plot(rows[finite], self.flux_1d[finite], color="C0", lw=0.9, label="Flux")
        if err is not None and np.any(err > 0):
            ax.fill_between(rows[finite],
                            (self.flux_1d - err)[finite],
                            (self.flux_1d + err)[finite],
                            color="C0", alpha=0.25, label="±1σ")
        ax.set_xlabel("Row (wavelength pixel)")
        ax.set_ylabel("Flux")
        ax.legend(loc="upper right")
        ax.set_title("Extracted 1D Spectrum")
        fig.tight_layout(pad=0.8)

        canvas = FigureCanvasTkAgg(fig, master=win)
        self._display_canvas_1d = canvas
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def _on_click_1d(event):
            if toolbar.mode or event.inaxes != ax:
                return
            row = int(round(event.xdata))
            n_rows = len(self.flux_1d)
            if 0 <= row < n_rows:
                self._go_to_row(row)
                self.lift()

        canvas.mpl_connect("button_press_event", _on_click_1d)
        canvas.draw()

    def _save_1d_fits(self):
        if self.flux_1d is None or not np.any(np.isfinite(self.flux_1d)):
            messagebox.showwarning("No data", "Run extraction first.")
            return
        if not HAS_FITS:
            messagebox.showerror("Missing dependency", "astropy is required.")
            return
        n_rows = len(self.flux_1d)
        n_finite = int(np.sum(np.isfinite(self.flux_1d)))
        missing = [i for i, v in enumerate(self.flux_1d) if not np.isfinite(v)]
        if n_finite < n_rows:
            unflagged = [i for i in missing
                         if self.row_flag is None or not self.row_flag[i]]
            if unflagged:
                if not self._confirm_incomplete_save(n_finite, n_rows, unflagged):
                    return
        if missing and self.row_flag is not None:
            for i in missing:
                self.row_flag[i] = True
            self._update_flag_btn()
            self._update_flag_overlay()
        base = os.path.splitext(os.path.basename(self.fits_path))[0]
        out_dir = os.path.dirname(self.fits_path)
        path = filedialog.asksaveasfilename(
            title="Save 1D spectrum",
            initialdir=out_dir,
            initialfile=f"{base}_1d.fits",
            defaultextension=".fits",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            hdu0 = _fits.PrimaryHDU(data=self.flux_1d, header=self.fits_header)
            hdu0.header["BUNIT"] = "flux"
            hdu0.header["EXTNAME"] = "FLUX"
            err = self.err_1d if self.err_1d is not None else np.full(n_rows, np.nan)
            hdu1 = _fits.ImageHDU(data=err, name="ERR")
            _fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)
            self._status_var.set(f"1D spectrum saved: {path}")
            messagebox.showinfo("Saved", f"1D spectrum written to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))
