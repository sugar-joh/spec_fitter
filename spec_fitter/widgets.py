"""
widgets.py — Reusable tkinter dialog helpers.
"""

import tkinter as tk
from tkinter import ttk


def _ask_extension(parent, labels: list, indices: list):
    """Show a modal dialog listing FITS extensions; return the chosen index or None."""
    dlg = tk.Toplevel(parent)
    dlg.title("Select FITS extension")
    dlg.resizable(False, False)
    dlg.grab_set()

    ttk.Label(dlg, text="This file has multiple 2D image extensions.\nSelect one to load:",
              padding=(12, 10, 12, 4)).pack(anchor=tk.W)

    var = tk.IntVar(value=indices[0])
    for label, idx in zip(labels, indices):
        ttk.Radiobutton(dlg, text=label, variable=var, value=idx).pack(
            anchor=tk.W, padx=16, pady=2)

    result = [None]

    def _ok():
        result[0] = var.get()
        dlg.destroy()

    def _cancel():
        dlg.destroy()

    btn_frame = ttk.Frame(dlg, padding=(12, 8))
    btn_frame.pack(fill=tk.X)
    ttk.Button(btn_frame, text="OK", command=_ok).pack(side=tk.RIGHT, padx=4)
    ttk.Button(btn_frame, text="Cancel", command=_cancel).pack(side=tk.RIGHT)

    dlg.protocol("WM_DELETE_WINDOW", _cancel)
    parent.wait_window(dlg)
    return result[0]
