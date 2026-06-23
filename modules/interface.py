"""
modules/interface.py
====================
Tkinter-based UI describing the 2-D slice viewer.
Delegates rendering to modules.renderer.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Optional

# Import the separated rendering engine
from modules.renderer import RingBuffer

log = logging.getLogger("interface")

# ── constants ────────────────────────────────────────────────────────────────
MIN_WIN_W       = 900
MIN_WIN_H       = 600
RESIZE_DEBOUNCE = 150         
RESIZE_THRESHOLD = 5          


class SeriesViewer:
    _STATIC_ROWS: list[tuple[str, str]] = [
        ("Patient Name",    "PatientName"),
        ("Patient ID",      "PatientID"),
        ("Date of Birth",   "PatientBirthDate"),
        ("Sex",             "PatientSex"),
        ("",                ""),                          
        ("Modality",        "Modality"),
        ("Manufacturer",    "Manufacturer"),
        ("Institution",     "InstitutionName"),
        ("",                ""),
        ("Study UID",       "StudyInstanceUID"),
        ("Series UID",      "SeriesInstanceUID"),
        ("",                ""),
        ("Rows × Columns",  "__rows_cols__"),
        ("Pixel Spacing",   "__pixel_spacing__"),
        ("Slice Thickness", "SliceThickness"),
        ("Slice Spacing",   "SliceSpacingComputed"),
        ("Total Slices",    "NumberOfSlices"),
    ]

    def __init__(self,
                 root:       tk.Misc,
                 series_uid: str,
                 entry:      dict,
                 on_close:   callable) -> None:

        self._root      = root
        self._uid       = series_uid
        self._volume    = entry["volume"]
        self._meta      = entry["metadata"]
        self._on_close  = on_close

        self._n_slices  = self._volume.shape[0]
        self._current_z = 0

        # Initialize RingBuffer from renderer module
        self._ring = RingBuffer()

        self._canvas_w = 1
        self._canvas_h = 1
        self._resize_after_id: Optional[str] = None
        self._last_canvas_w   = 0
        self._last_canvas_h   = 0

        self._iid_current_slice: Optional[str] = None
        self._iid_current_z:     Optional[str] = None

        self._build_window()

    def _build_window(self) -> None:
        modality   = self._meta.get("Modality") or "unknown"
        patient    = self._meta.get("PatientName") or "Unknown Patient"
        title      = f"{modality} — {patient} — {self._uid[-12:]}"

        self._win = tk.Toplevel(self._root)
        self._win.title(title)
        self._win.minsize(MIN_WIN_W, MIN_WIN_H)
        self._win.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self._pane = ttk.PanedWindow(self._win, orient=tk.HORIZONTAL)
        self._pane.pack(fill=tk.BOTH, expand=True)

        self._left_frame = ttk.Frame(self._pane, width=270)
        self._left_frame.pack_propagate(False)
        self._pane.add(self._left_frame, weight=30)
        self._build_metadata_panel()

        self._right_frame = ttk.Frame(self._pane)
        self._pane.add(self._right_frame, weight=70)
        self._build_image_panel()

        self._win.after(50, self._initial_build)

    def _build_metadata_panel(self) -> None:
        lbl = ttk.Label(self._left_frame, text="Series Metadata", font=("TkDefaultFont", 10, "bold"))
        lbl.pack(anchor=tk.W, padx=6, pady=(6, 2))

        cols = ("field", "value")
        tv   = ttk.Treeview(self._left_frame, columns=cols, show="headings", selectmode="none")
        tv.heading("field", text="Field")
        tv.heading("value", text="Value")
        tv.column("field", width=120, stretch=False)
        tv.column("value", width=140, stretch=True)

        sb = ttk.Scrollbar(self._left_frame, orient=tk.VERTICAL, command=tv.yview)
        tv.configure(yscrollcommand=sb.set)

        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=4)
        tv.pack(fill=tk.BOTH, expand=True, padx=(6, 0), pady=4)

        self._tv = tv
        self._populate_metadata_table()

    def _populate_metadata_table(self) -> None:
        meta = self._meta
        tv   = self._tv

        def fmt(val) -> str:
            return "N/A" if val is None else str(val)

        for label, key in self._STATIC_ROWS:
            if label == "":
                tv.insert("", tk.END, values=("──────────", "──────────"))
                continue

            if key == "__rows_cols__":
                value = f"{meta.get('Rows', '?')} × {meta.get('Columns', '?')}"
            elif key == "__pixel_spacing__":
                ps = meta.get("PixelSpacing")
                value = f"{ps[0]:.4f} × {ps[1]:.4f} mm" if ps else "N/A"
            elif key in ("StudyInstanceUID", "SeriesInstanceUID"):
                raw = fmt(meta.get(key))
                value = raw if len(raw) <= 20 else f"…{raw[-18:]}"
            elif key in ("SliceThickness", "SliceSpacingComputed"):
                raw = meta.get(key)
                value = f"{raw:.4f} mm" if raw is not None else "N/A"
            else:
                value = fmt(meta.get(key))

            tv.insert("", tk.END, values=(label, value))

        tv.insert("", tk.END, values=("──────────", "──────────"))

        self._iid_current_slice = tv.insert("", tk.END, values=("Current Slice", "─"))
        self._iid_current_z     = tv.insert("", tk.END, values=("Current Z",     "─"))
        self._update_live_rows()

    def _update_live_rows(self) -> None:
        z_pos_list = self._meta.get("ZPositions") or []
        n          = self._meta.get("NumberOfSlices") or self._n_slices
        real_z = min(self._current_z, len(z_pos_list) - 1)
        z_mm   = z_pos_list[real_z] if real_z >= 0 else 0.0

        if self._iid_current_slice:
            self._tv.item(self._iid_current_slice, values=("Current Slice", f"{self._current_z + 1} / {n}"))
        if self._iid_current_z:
            self._tv.item(self._iid_current_z, values=("Current Z", f"{z_mm:.3f} mm"))

    def _build_image_panel(self) -> None:
        self._canvas = tk.Canvas(self._right_frame, bg="black", cursor="crosshair", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._img_id = self._canvas.create_image(0, 0, anchor=tk.NW)

        self._status_var = tk.StringVar(value="Loading…")
        status_bar = ttk.Label(self._right_frame, textvariable=self._status_var, anchor=tk.W, relief=tk.SUNKEN, padding=(4, 2))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._canvas.bind("<MouseWheel>", self._on_scroll)   
        self._canvas.bind("<Button-4>",   self._on_scroll)   
        self._canvas.bind("<Button-5>",   self._on_scroll)   
        self._right_frame.bind("<Configure>", self._on_resize)

    def _update_status(self) -> None:
        z_pos_list = self._meta.get("ZPositions") or []
        n          = self._meta.get("NumberOfSlices") or self._n_slices
        real_z     = min(self._current_z, len(z_pos_list) - 1)
        z_mm       = z_pos_list[real_z] if real_z >= 0 else 0.0
        self._status_var.set(f"Slice {self._current_z + 1} / {n}    Z = {z_mm:.3f} mm")

    def _initial_build(self) -> None:
        self._refresh_canvas_size()
        if self._canvas_w < 2 or self._canvas_h < 2:
            self._win.after(50, self._initial_build)
            return
        self._rebuild_ring()
        self._display_current()

    def _refresh_canvas_size(self) -> None:
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w > 1 and h > 1:
            self._canvas_w = w
            self._canvas_h = h

    def _rebuild_ring(self) -> None:
        self._ring.build(
            volume    = self._volume,
            current_z = self._current_z,
            canvas_h  = self._canvas_h,
            canvas_w  = self._canvas_w,
        )

    def _display_current(self) -> None:
        photo = self._ring.current_photo()
        if photo is not None:
            self._canvas.itemconfig(self._img_id, image=photo)
            self._canvas.image = photo  # type: ignore[attr-defined]
        self._update_status()
        self._update_live_rows()

    def _on_scroll(self, event: tk.Event) -> None:
        if event.num == 4:
            direction = +1
        elif event.num == 5:
            direction = -1
        else:
            direction = +1 if event.delta > 0 else -1

        moved = self._ring.advance(
            direction = direction,
            volume    = self._volume,
            canvas_h  = self._canvas_h,
            canvas_w  = self._canvas_w,
        )

        if moved:
            self._current_z = self._ring.z_index[self._ring.head]
            self._display_current()

    def _on_resize(self, event: tk.Event) -> None:
        if self._resize_after_id is not None:
            self._win.after_cancel(self._resize_after_id)
        self._resize_after_id = self._win.after(RESIZE_DEBOUNCE, self._handle_resize)

    def _handle_resize(self) -> None:
        self._resize_after_id = None
        self._refresh_canvas_size()

        w, h = self._canvas_w, self._canvas_h
        if (abs(w - self._last_canvas_w) < RESIZE_THRESHOLD and
                abs(h - self._last_canvas_h) < RESIZE_THRESHOLD):
            return

        self._last_canvas_w = w
        self._last_canvas_h = h
        self._rebuild_ring()
        self._display_current()

    def _on_window_close(self) -> None:
        log.info("Viewer closed for series %s", self._uid)
        self._win.destroy()
        self._on_close()

def show_viewer(series_dict: dict[str, dict]) -> None:
    if not series_dict:
        log.warning("show_viewer: series_dict is empty — nothing to display")
        return

    queue: list[tuple[str, dict]] = list(series_dict.items())
    log.info("show_viewer: %d series in queue", len(queue))

    root = tk.Tk()
    root.withdraw()

    def open_next() -> None:
        if not queue:
            log.info("show_viewer: all series shown — destroying root")
            root.destroy()
            return

        uid, entry = queue.pop(0)
        SeriesViewer(
            root       = root,
            series_uid = uid,
            entry      = entry,
            on_close   = open_next,
        )

    open_next()
    root.mainloop()
