# nbs_overlay_clicker_fixed.py
# Optimized + fixes for: heavy-grid lag (e.g. 76x56) and clicking while overlay hidden
# - Precomputes tick screen coordinates (homography + skew) once when grid/settings change
# - Caches tick_positions so play loop does O(1) lookups when clicking delay cells
# - Redraws overlay only when parameters actually change (overlay_dirty flag)
# - Overlay can be hidden but mapping remains available for playback
# - Minor micro-optimizations in update loop to avoid repeated expensive work

import pynbs
import tkinter as tk
from tkinter import filedialog, scrolledtext
import ctypes
from ctypes import wintypes
import threading
import keyboard
import math
import numpy as np
import time
import random
# ---------------- Windows constants ----------------
GWL_EXSTYLE = -20
WS_EX_LAYERED     = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW  = 0x00000080

# SendInput mouse flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

# ---------------- Music constants ----------------
NOTE_NAMES = [
    "F#0", "G1", "G#2", "A3", "A#4", "B5",
    "C6", "C#7", "D8", "D#9", "E10", "F11",
    "F#12", "G13", "G#14", "A15", "A#16", "B17",
    "C18", "C#19", "D20", "D#21", "E22", "F23", "F#24"
]
NOTE_MIN = 33
NOTE_MAX = 57

# ---------------- Globals ----------------
# Note positions and required counts
note_positions = {note: [] for note in NOTE_NAMES}   # captured positions per note name
required_counts = {note: 0 for note in NOTE_NAMES}

# Delay grid configuration
delay_grid = {"top_left": None, "bottom_right": None}

# Active (non-empty) ticks compression mapping:
# - active_ticks: list of original tick indices that contain at least one note
# - active_tick_index: maps original tick -> compressed index in active_ticks
active_ticks = []
active_tick_index = {}

# Tempo tool support
# - tempo_tool_pos: screen position of the in-game tempo input/apply UI
tempo_tool_pos = None

# Armed note (single): name of the note currently armed, or None
armed_note = None

# Lock to protect armed_note and note_positions changes
armed_lock = threading.Lock()

# Flag used to indicate we're currently waiting for a grid corner (so armed handler ignores F6)
capturing_grid_corner = False

# Overlay-related globals
overlay_window = None
overlay_canvas = None
overlay_transparent_supported = False
song_ticks = 0
overlay_shown = False
columns_override = None
rows_override = None
notes_by_tick = {}
tempo = 0.0
stop_play = False
last_overlay_mapping = None
# Precomputed tick absolute positions (len = song_ticks) or None
tick_positions = None
# Dirty flag: set True when mapping/visuals need redraw
overlay_dirty = True
# Keep last-drawn params to avoid redrawing unchanged overlay
_last_drawn_params = None

# ---------------- Helpers ----------------
def get_cursor_pos():
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def key_to_note_name(key):
    if NOTE_MIN <= key <= NOTE_MAX:
        return NOTE_NAMES[key - NOTE_MIN]
    return f"Unknown({key})"

def instrument_to_name(inst):
    return {0:"Piano",1:"Bass Drum",2:"Snare Drum",3:"Click",
            4:"Guitar",5:"Flute",6:"Bell",7:"Chime",
            8:"Xylophone",9:"Iron Xylophone",10:"Cow Bell",
            11:"Didgeridoo",12:"Bit",13:"Banjo",14:"Pling"}.get(inst, f"Unknown({inst})")


def quantize_delay_to_step(value, step=0.01, minimum=0.05):
    """Quantize a delay (seconds) to the given step using current round mode.

    - step: game resolution (0.01 seconds)
    - minimum: smallest allowed value in game (0.05 seconds)
    Mode is controlled by round_mode_var ("up" or "down").
    """
    # Non-positive delays don't need a block
    if value <= 0:
        return 0.0
    try:
        mode = round_mode_var.get()
    except Exception:
        mode = "down"
    if step <= 0:
        q_val = value
    else:
        ratio = value / step
        if mode == "up":
            q = math.ceil(ratio)
        else:
            q = math.floor(ratio)
        q_val = q * step
    # Enforce minimum if any delay is needed
    if 0.0 < q_val < minimum:
        q_val = minimum
    return max(0.0, q_val)

def make_window_clickthrough(tk_window):
    try:
        hwnd = wintypes.HWND(ctypes.windll.user32.GetParent(tk_window.winfo_id()))
        if not hwnd:
            hwnd = wintypes.HWND(tk_window.winfo_id())
        ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
    except Exception:
        pass

def get_screen_size():
    sx = ctypes.windll.user32.GetSystemMetrics(0)
    sy = ctypes.windll.user32.GetSystemMetrics(1)
    return sx, sy

def to_absolute_coords(x, y):
    sx, sy = get_screen_size()
    if sx <= 1 or sy <= 1:
        return 0, 0
    ax = int(round(x * 65535.0 / (sx - 1)))
    ay = int(round(y * 65535.0 / (sy - 1)))
    return ax, ay

# ---------------- Layout helpers ----------------
def compute_best_columns(width, height, ticks):
    if ticks <= 0:
        return 1
    max_test = min(ticks, 2000)
    best = 1
    best_score = float("inf")
    for cols in range(1, max_test + 1):
        rows = math.ceil(ticks / cols)
        cell_ratio = (width / cols) / (height / max(rows, 1))
        score = abs(cell_ratio - 1.0)
        if ticks % cols == 0:
            score *= 0.5
        score += 0.00001 * cols
        if score < best_score:
            best_score = score
            best = cols
    return best

def compute_best_rows(width, height, ticks):
    if ticks <= 0:
        return 1
    max_test = min(ticks, 2000)
    best = 1
    best_score = float("inf")
    for rows in range(1, max_test + 1):
        cols = math.ceil(ticks / rows)
        cell_ratio = (width / max(cols,1)) / (height / rows)
        score = abs(cell_ratio - 1.0)
        if ticks % rows == 0:
            score *= 0.5
        score += 0.00001 * rows
        if score < best_score:
            best_score = score
            best = rows
    return best

# ---------------- Homography ----------------
def compute_homography(src_pts, dst_pts):
    A = []
    b = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
        b.append(xp)
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
        b.append(yp)
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    try:
        h = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1.0]], dtype=np.float64)
    return H

def apply_homography(H, x, y):
    vec = np.array([x, y, 1.0], dtype=np.float64)
    tp = H.dot(vec)
    if abs(tp[2]) < 1e-9:
        return float(tp[0]), float(tp[1])
    return float(tp[0] / tp[2]), float(tp[1] / tp[2])

# ---------------- Required counts ----------------
def compute_required_counts():
    global required_counts
    req = {note: 0 for note in NOTE_NAMES}
    for t in range(song_ticks):
        counter = {}
        for n in notes_by_tick.get(t, []):
            name = key_to_note_name(n.key)
            counter[name] = counter.get(name, 0) + 1
        for name, c in counter.items():
            if c > req.get(name, 0):
                req[name] = c
    required_counts = req
    try:
        for note, btn in note_buttons.items():
            have = len(note_positions.get(note, []))
            need = required_counts.get(note, 0)
            if need > 0:
                btn.config(text=f"{note}: {have}/{need}")
            else:
                btn.config(text=f"{note}: {have}")
    except Exception:
        pass

# ---------------- Mapping / caching (NEW) ----------------
def update_mapping():
    """Compute and cache the overlay homography, grid metrics and absolute tick positions.
    Uses *compressed* tick indices: only ticks that actually contain notes (active_ticks)
    receive delay cells / red dots. This makes the grid smaller and faster.
    """
    global last_overlay_mapping, tick_positions, overlay_dirty, active_ticks

    tl = delay_grid.get("top_left"); br = delay_grid.get("bottom_right")
    # If we have no grid or no song or no active ticks, clear mapping
    if not tl or not br or song_ticks <= 0 or not active_ticks:
        last_overlay_mapping = None
        tick_positions = None
        overlay_dirty = True
        return

    # Number of delay cells = number of non-empty ticks
    effective_ticks = len(active_ticks)

    x1, y1 = tl; x2, y2 = br
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    width = max(1, x2 - x1); height = max(1, y2 - y1)

    # Use compressed tick count for grid shape instead of full song_ticks
    cols = columns_override if columns_override is not None else compute_best_columns(width, height, effective_ticks)
    rows = rows_override if rows_override is not None else compute_best_rows(width, height, effective_ticks)

    # Build homography from local overlay coords to screen-space quad
    tl_local = (0.0, 0.0)
    tr_local = (float(width), 0.0)
    br_local = (float(width), float(height))
    bl_local = (0.0, float(height))
    src_rect = [tl_local, tr_local, br_local, bl_local]
    dst_quad = [ (tl[0] - x1, tl[1] - y1), (br[0] - x1, tl[1] - y1), (br[0] - x1, br[1] - y1), (tl[0] - x1, br[1] - y1) ]
    H = compute_homography(src_rect, dst_quad)

    # get origin screen coordinates (top-left of bounding rect)
    origin_x = x1; origin_y = y1
    cell_w = width / max(cols,1)
    cell_h = height / max(rows,1)

    # Read skew once
    try:
        skew_x = float(skew_x_var.get())
        skew_y = float(skew_y_var.get())
    except Exception:
        skew_x = 0.0; skew_y = 0.0

    # One absolute screen position per *active* tick (non-empty)
    positions = [None] * effective_ticks
    sx_screen, sy_screen = get_screen_size()
    center_x = width / 2.0
    center_y = height / 2.0

    # Place only active (non-empty) ticks into the grid sequentially
    for i, tick in enumerate(active_ticks):
        col = i % cols
        row = i // cols
        sx = (col + 0.5) * cell_w
        sy = (row + 0.5) * cell_h
        x_local, y_local = apply_homography(H, sx, sy)

        # Apply pixel-based skew (same logic as overlay)
        if center_x != 0:
            dx = (x_local - center_x) / center_x
        else:
            dx = 0.0
        if center_y != 0:
            dy = (y_local - center_y) / center_y
        else:
            dy = 0.0
        x_local = x_local + dx * skew_x * center_x
        y_local = y_local + dy * skew_y * center_y

        abs_x = origin_x + x_local
        abs_y = origin_y + y_local

        # clamp to primary screen
        abs_x = max(0.0, min(abs_x, sx_screen - 1.0))
        abs_y = max(0.0, min(abs_y, sy_screen - 1.0))
        positions[i] = (abs_x, abs_y)

    last_overlay_mapping = {
        'H': H,
        'cols': cols,
        'rows': rows,
        'cell_w': cell_w,
        'cell_h': cell_h,
        'origin_x': origin_x,
        'origin_y': origin_y,
        'width': width,
        'height': height,
        'min_x': x1,
        'min_y': y1,
        'count': effective_ticks,
    }
    tick_positions = positions
    overlay_dirty = True

# ---------------- NBS loader ----------------
def load_nbs():
    global song_ticks, notes_by_tick, tempo, active_ticks, active_tick_index
    file_path = filedialog.askopenfilename(filetypes=[("NBS files", "*.nbs")])
    if not file_path:
        return
    song = pynbs.read(file_path)
    try:
        tempo = float(song.header.tempo)
    except Exception:
        tempo = 0.0
    try:
        header_length = int(song.header.song_length)
    except Exception:
        header_length = 0

    notes_by_tick.clear()
    max_tick = 0
    for n in song.notes:
        if NOTE_MIN <= n.key <= NOTE_MAX:
            notes_by_tick.setdefault(n.tick, []).append(n)
            if n.tick > max_tick:
                max_tick = n.tick

    # ✅ Fix: some NBS files set header_length wrong, so include last tick
    song_ticks = max(header_length, max_tick + 1)

    # Build compressed tick mapping: only ticks that actually contain notes
    active_ticks = [t for t in range(song_ticks) if notes_by_tick.get(t)]
    active_tick_index = {t: i for i, t in enumerate(active_ticks)}

    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"Tempo: {tempo:.2f} ticks/sec\n")
    output_text.insert(tk.END, f"Header ticks: {header_length}, derived ticks: {song_ticks}\n")
    # Inform user how many delay blocks they actually need after compression
    output_text.insert(tk.END, f"Compressed non-empty ticks: {len(active_ticks)} (you need this many delay blocks)\n")
    output_text.insert(tk.END, "Tick | Delay(s) | Notes (Layer, Instrument, NoteName)\\n")

    prev_tick = 0
    for tick in range(song_ticks):
        notes_in_tick = notes_by_tick.get(tick, [])
        delay_sec = (tick - prev_tick) / tempo if tempo > 0 else 0.0
        if notes_in_tick:
            notes_str = ", ".join([
                f"(L{n.layer}, {instrument_to_name(n.instrument)}, {key_to_note_name(n.key)})"
                for n in notes_in_tick
            ])
        else:
            notes_str = "<no notes>"
        output_text.insert(tk.END, f"{tick:3d} | {delay_sec:7.4f} | {notes_str}\n")
        prev_tick = tick

    compute_required_counts()

    # update mapping regardless of overlay visibility (this is the key: mapping exists while overlay hidden)
    update_mapping()
    # update overlay if it's visible
    root.after(0, update_or_create_overlay)

# ---------------- Capture handlers ----------------
def capture_note_position(note):
    """
    Toggle armed state for a note.
    - If note was not armed: arm it (disarm any previous), update label to 'Armed ...'
    - If note was armed: disarm it and update label back to have/need
    Press F6 to capture while the note is armed.
    """
    global armed_note
    prev = None
    with armed_lock:
        if armed_note == note:
            # disarm
            armed_note = None
            prev = None
        else:
            prev = armed_note
            armed_note = note

    # update UI for previous (disarm) and current (armed or normal)
    def _update_labels():
        # previous note (if exists) -> set normal label
        if prev and prev in note_buttons:
            have_prev = len(note_positions.get(prev, []))
            need_prev = required_counts.get(prev, 0)
            if need_prev > 0:
                note_buttons[prev].config(text=f"{prev}: {have_prev}/{need_prev}")
            else:
                note_buttons[prev].config(text=f"{prev}: {have_prev}")

        # current note
        have = len(note_positions.get(note, []))
        need = required_counts.get(note, 0)
        # If we just armed it:
        with armed_lock:
            is_armed = (armed_note == note)
        if is_armed:
            if need > 0:
                note_buttons[note].config(text=f"{note}: Armed {have}/{need}")
            else:
                note_buttons[note].config(text=f"{note}: Armed {have}")
            output_text.insert(tk.END, f"{note} armed. Press F6 to capture positions; click note to cancel.\n")
        else:
            if need > 0:
                note_buttons[note].config(text=f"{note}: {have}/{need}")
            else:
                note_buttons[note].config(text=f"{note}: {have}")
            output_text.insert(tk.END, f"{note} disarmed.\n")

    root.after(0, _update_labels)


def capture_grid_corner(corner):
    btn = corner_buttons[corner]
    btn.config(text=f"{corner.replace('_',' ').title()}: Waiting...")
    def wait_f6():
        global capturing_grid_corner
        capturing_grid_corner = True
        keyboard.wait("f6")
        x, y = get_cursor_pos()
        delay_grid[corner] = (x, y)
        capturing_grid_corner = False
        root.after(0, lambda: btn.config(text=f"{corner.replace('_',' ').title()}: ({x},{y})"))
        # mapping changed -> update mapping (precompute positions)
        update_mapping()
        root.after(0, update_or_create_overlay)
    threading.Thread(target=wait_f6, daemon=True).start()

def _f6_global_handler(evt):
    global armed_note  # must be first line

    if capturing_grid_corner:  # don't interfere if corner capture active
        return

    with armed_lock:
        note = armed_note  # safe to read/write

    if not note:
        return

    x, y = get_cursor_pos()
    with armed_lock:
        positions = note_positions.setdefault(note, [])
        need = required_counts.get(note, 0)
        have = len(positions)

        if need > 0 and have >= need:
            armed_note = None  # auto-disarm
            root.after(0, lambda: note_buttons[note].config(text=f"{note}: {have}/{need}"))
            root.after(0, lambda: output_text.insert(tk.END, f"Note {note} already has required captures ({have}). Auto-disarmed.\n"))
            return

        positions.append((x, y))
        have += 1

    # Update UI back in the main thread
    def _ui_updates(x=x, y=y, note=note, have=have, need=need):
        global armed_note  # Move this to the top of the function
        
        # If limit reached -> disarm and show normal label
        if need > 0 and have >= need:
            # Limit reached
            try:
                note_buttons[note].config(text=f"{note}: {have}/{need}")
            except Exception:
                pass
            output_text.insert(tk.END, f"Captured {note} at ({x},{y}) — total {have}. Reached required captures; auto-disarmed.\n")
            # Ensure armed state is reset
            with armed_lock:
                if armed_note == note:
                    armed_note = None
        else:
            # Still armed (or no limit)
            try:
                if need > 0:
                    note_buttons[note].config(text=f"{note}: Armed {have}/{need}")
                else:
                    note_buttons[note].config(text=f"{note}: Armed {have}")
            except Exception:
                pass
            output_text.insert(tk.END, f"Captured {note} at ({x},{y}) - total {have}\n")

    root.after(0, _ui_updates)

# ---------------- Overlay ----------------
def create_overlay(preview=False):
    global overlay_window, overlay_canvas, overlay_transparent_supported, _last_drawn_params
    tl = delay_grid.get("top_left"); br = delay_grid.get("bottom_right")
    if not tl or not br or song_ticks <= 0:
        destroy_overlay(); return

    pts = [p for p in delay_grid.values() if p]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    min_x, max_x = int(min(xs)), int(max(xs))
    min_y, max_y = int(min(ys)), int(max(ys))
    width = max(1, max_x - min_x); height = max(1, max_y - min_y)

    try:
        if overlay_window is not None:
            overlay_window.destroy()
    except Exception:
        pass

    overlay_window = tk.Toplevel(root)
    overlay_window.overrideredirect(True)
    overlay_window.attributes("-topmost", True)
    overlay_window.geometry(f"{width}x{height}+{min_x}+{min_y}")

    transparent_color = "magenta"
    overlay_canvas = tk.Canvas(overlay_window, width=width, height=height, bg=transparent_color, highlightthickness=0)
    overlay_canvas.pack(fill="both", expand=True)

    overlay_transparent_supported = False
    if not preview:
        try:
            overlay_window.attributes("-transparentcolor", transparent_color)
            overlay_transparent_supported = True
        except Exception:
            overlay_transparent_supported = False

    if not overlay_transparent_supported:
        try:
            overlay_window.attributes("-alpha", 0.35)
        except Exception:
            pass

    root.after(50, lambda: make_window_clickthrough(overlay_window))

    try:
        overlay_window.lift(); overlay_window.update_idletasks()
    except Exception:
        pass

    # force overlay redraw next time
    global overlay_dirty
    overlay_dirty = True
    _last_drawn_params = None
    update_overlay()

def destroy_overlay():
    global overlay_window, overlay_canvas, _last_drawn_params
    try:
        if overlay_window is not None:
            overlay_window.destroy()
    except Exception:
        pass
    overlay_window = None
    overlay_canvas = None
    _last_drawn_params = None

def update_overlay():
    """Redraw overlay only when mapping or visual parameters changed.
    Uses tick_positions precomputed by update_mapping() if available.
    """
    global _last_drawn_params, overlay_dirty
    if overlay_canvas is None:
        return
    if last_overlay_mapping is None:
        return

    # collect current visual params to detect change
    try:
        user_dot_size = int(dot_size_var.get())
        if user_dot_size < 1:
            user_dot_size = None
    except Exception:
        user_dot_size = None

    try:
        skew_x = float(skew_x_var.get())
        skew_y = float(skew_y_var.get())
    except Exception:
        skew_x = 0.0; skew_y = 0.0

    params = (
        last_overlay_mapping['cols'], last_overlay_mapping['rows'],
        last_overlay_mapping['width'], last_overlay_mapping['height'],
        user_dot_size, skew_x, skew_y, overlay_transparent_supported
    )

    # If nothing changed and overlay is not marked dirty, skip redraw
    if not overlay_dirty and params == _last_drawn_params:
        return

    _last_drawn_params = params

    width = last_overlay_mapping['width']
    height = last_overlay_mapping['height']
    min_x = last_overlay_mapping['min_x']
    min_y = last_overlay_mapping['min_y']

    overlay_canvas.delete("all")

    min_dim = min(last_overlay_mapping['cell_w'], last_overlay_mapping['cell_h'])
    if user_dot_size is not None:
        dot_r = user_dot_size
    else:
        dot_r = 1 if min_dim < 8 else 2 if min_dim < 20 else 3

    color = "red" if overlay_transparent_supported else "white"

    # if we have precomputed positions, draw them relative to overlay
    if tick_positions is not None:
        # draw only visible ticks within overlay bounds
        for i, (abs_x, abs_y) in enumerate(tick_positions):
            x_local = abs_x - min_x
            y_local = abs_y - min_y
            if not (math.isfinite(x_local) and math.isfinite(y_local)):
                continue
            if x_local < -dot_r or x_local > width + dot_r or y_local < -dot_r or y_local > height + dot_r:
                continue
            x1 = int(round(x_local - dot_r)); y1 = int(round(y_local - dot_r))
            x2 = int(round(x_local + dot_r)); y2 = int(round(y_local + dot_r))
            # create simple oval; avoid extra options
            overlay_canvas.create_oval(x1, y1, x2, y2, fill=color, outline="")
    else:
        # fallback: draw by computing coords on the fly (rare)
        cols = last_overlay_mapping['cols']; rows = last_overlay_mapping['rows']
        cell_w = last_overlay_mapping['cell_w']; cell_h = last_overlay_mapping['cell_h']
        H = last_overlay_mapping['H']
        origin_x = last_overlay_mapping['origin_x']; origin_y = last_overlay_mapping['origin_y']
        for i in range(song_ticks):
            col = i % cols
            row = i // cols
            sx = (col + 0.5) * cell_w
            sy = (row + 0.5) * cell_h
            x_local, y_local = apply_homography(H, sx, sy)
            # pixel-based skew
            center_x = width / 2.0; center_y = height / 2.0
            dx = (x_local - center_x) / center_x if center_x != 0 else 0.0
            dy = (y_local - center_y) / center_y if center_y != 0 else 0.0
            try:
                skew_x = float(skew_x_var.get()); skew_y = float(skew_y_var.get())
            except Exception:
                skew_x = 0.0; skew_y = 0.0
            x_local = x_local + dx * skew_x * center_x
            y_local = y_local + dy * skew_y * center_y
            x1 = int(round(x_local - dot_r)); y1 = int(round(y_local - dot_r))
            x2 = int(round(x_local + dot_r)); y2 = int(round(y_local + dot_r))
            overlay_canvas.create_oval(x1, y1, x2, y2, fill=color, outline="")

    overlay_dirty = False

# update overlay periodically if shown
def update_or_create_overlay():
    if overlay_shown:
        if overlay_window is None:
            create_overlay()
        else:
            update_overlay()

# ---------------- Overlay controls ----------------
def set_columns_from_entry():
    global columns_override
    val = cols_entry.get().strip()
    if val == "":
        columns_override = None
    else:
        try:
            c = int(val)
            if c < 1:
                c = None
            columns_override = c
        except Exception:
            columns_override = None
    # mapping changed
    update_mapping()
    update_or_create_overlay()

def set_rows_from_entry():
    global rows_override
    val = rows_entry.get().strip()
    if val == "":
        rows_override = None
    else:
        try:
            r = int(val)
            if r < 1:
                r = None
            rows_override = r
        except Exception:
            rows_override = None
    update_mapping()
    update_or_create_overlay()

def apply_grid_settings():
    set_columns_from_entry(); set_rows_from_entry()
    try: overlay_status_var.set("Overlay: updated grid")
    except Exception: pass

def toggle_overlay():
    global overlay_shown
    overlay_shown = not overlay_shown
    if overlay_shown:
        create_overlay(preview=False)
        show_grid_btn.config(text="Hide Grid")
        try: overlay_status_var.set("Overlay: shown")
        except Exception: pass
    else:
        destroy_overlay()
        show_grid_btn.config(text="Show Grid")
        try: overlay_status_var.set("Overlay: hidden")
        except Exception: pass

# ---------------- SendInput wrapper (mouse) ----------------
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p)
    ]

class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("i",)
    _fields_ = [("type", ctypes.c_ulong), ("i", _I)]

SendInput = ctypes.windll.user32.SendInput

def _send_input_mouse(ax, ay, flags):
    inp = INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.mi.dx = int(ax)
    inp.mi.dy = int(ay)
    inp.mi.mouseData = 0
    inp.mi.dwFlags = int(flags)
    inp.mi.time = 0
    inp.mi.dwExtraInfo = None
    SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

# keep move_smooth_to but use a fixed short duration (not used by teleport+wiggle flow)
def move_smooth_to(x_target, y_target):
    """Smooth interpolation to target using easing. Uses a short fixed duration.
    """
    total_s = 0.06

    x0, y0 = get_cursor_pos()
    dx = x_target - x0; dy = y_target - y0
    dist = math.hypot(dx, dy)
    if dist < 1:
        ax, ay = to_absolute_coords(x_target, y_target)
        _send_input_mouse(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        return

    steps = max(int(dist * 3), 12)
    per_step = total_s / steps
    for i in range(1, steps + 1):
        t = i / steps
        t_s = t * t * (3 - 2 * t)
        xi = x0 + dx * t_s
        yi = y0 + dy * t_s
        ax, ay = to_absolute_coords(xi, yi)
        _send_input_mouse(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        time.sleep(per_step)
    ax, ay = to_absolute_coords(x_target, y_target)
    _send_input_mouse(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)


# Modified: teleport + circular wiggle before click

def move_and_click_abs(x, y, pause_before_click=0.006):
    """Teleport instantly to (x,y), perform a small circular wiggle around it, then click.
    Wiggle amplitude and count are configurable via wiggle_amp_var and wiggle_count_var.
    """
    try:
        # immediate teleport to exact position (absolute coords)
        ax, ay = to_absolute_coords(x, y)
        _send_input_mouse(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        # tiny pause to let the OS update cursor
        time.sleep(0.002)

        # get user-configured wiggle params
        try:
            amp = float(wiggle_amp_var.get())
            cnt = int(wiggle_count_var.get())
            if amp < 0: amp = 0.0
            if cnt < 0: cnt = 0
        except Exception:
            amp = 2.0; cnt = 3

        # do circular wiggle: evenly spaced angles around the circle
        if cnt > 0 and amp > 0.0:
            for i in range(cnt):
                angle = 2.0 * math.pi * (i / cnt)
                # add a tiny random jitter to radius and angle so it's not perfectly mechanical
                r = amp * (0.7 + 0.3 * random.random())
                a = angle + random.uniform(-0.15, 0.15)
                ox = x + math.cos(a) * r
                oy = y + math.sin(a) * r
                ax, ay = to_absolute_coords(ox, oy)
                _send_input_mouse(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
                time.sleep(0.007)

        # final precise position before click
        ax, ay = to_absolute_coords(x, y)
        _send_input_mouse(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        time.sleep(pause_before_click)

        # click (left down/up)
        _send_input_mouse(0, 0, MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.010)
        _send_input_mouse(0, 0, MOUSEEVENTF_LEFTUP)
        time.sleep(0.004)
    except Exception:
        # fallback in case SendInput fails for any reason
        try:
            ctypes.windll.user32.SetCursorPos(int(round(x)), int(round(y)))
            # small circular wiggle fallback using SetCursorPos
            try:
                amp = float(wiggle_amp_var.get())
                cnt = int(wiggle_count_var.get())
            except Exception:
                amp = 2.0; cnt = 3
            if cnt > 0 and amp > 0.0:
                for i in range(cnt):
                    angle = 2.0 * math.pi * (i / cnt)
                    r = amp * (0.7 + 0.3 * random.random())
                    a = angle + random.uniform(-0.15, 0.15)
                    ox = int(round(x + math.cos(a) * r))
                    oy = int(round(y + math.sin(a) * r))
                    ctypes.windll.user32.SetCursorPos(ox, oy)
                    time.sleep(0.007)
            ctypes.windll.user32.SetCursorPos(int(round(x)), int(round(y)))
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        except Exception:
            pass


def compute_screen_coords_for_tick(tick_index):
    """Return screen coordinates for a *logical* tick index.

    tick_index is always the ORIGINAL NBS tick number. We map it through
    active_tick_index -> compressed index, then into tick_positions / grid.
    """
    global last_overlay_mapping, tick_positions, active_ticks, active_tick_index

    # Map original tick -> compressed index (only non-empty ticks have cells)
    idx = active_tick_index.get(tick_index)
    if idx is None:
        return None

    # If we precomputed tick_positions, just return the compressed index
    if tick_positions is not None:
        if 0 <= idx < len(tick_positions):
            return tick_positions[idx]
        return None

    # If we don't have an overlay mapping, fall back to a simple grid math
    if last_overlay_mapping is None:
        tl = delay_grid.get("top_left"); br = delay_grid.get("bottom_right")
        if not tl or not br:
            return None
        x1, y1 = tl; x2, y2 = br
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        width = max(1, x2 - x1); height = max(1, y2 - y1)
        cols = columns_override if columns_override is not None else 1
        eff = max(1, len(active_ticks))
        rows = rows_override if rows_override is not None else max(1, math.ceil(eff / cols)) if cols > 0 else 1
        col = idx % cols
        row = idx // cols
        cell_w = width / max(cols, 1)
        cell_h = height / max(rows, 1)
        cx = x1 + col * cell_w + cell_w / 2.0
        cy = y1 + row * cell_h + cell_h / 2.0
        return cx, cy

    # Use the overlay mapping (homography + same pixel-based skew as overlay)
    H = last_overlay_mapping['H']
    cols = last_overlay_mapping['cols']; rows = last_overlay_mapping['rows']
    cell_w = last_overlay_mapping['cell_w']; cell_h = last_overlay_mapping['cell_h']
    origin_x = last_overlay_mapping['origin_x']; origin_y = last_overlay_mapping['origin_y']
    width = last_overlay_mapping['width']; height = last_overlay_mapping['height']

    col = tick_index % cols
    row = tick_index // cols
    sx = (col + 0.5) * cell_w
    sy = (row + 0.5) * cell_h
    x_local, y_local = apply_homography(H, sx, sy)

    # Apply the SAME FIXED skew calculation as in update_overlay (pixel-based)
    try:
        skew_x = float(skew_x_var.get())
        skew_y = float(skew_y_var.get())
    except Exception:
        skew_x = 0.0
        skew_y = 0.0

    center_x = width / 2.0
    center_y = height / 2.0
    if center_x != 0:
        dx = (x_local - center_x) / center_x
    else:
        dx = 0.0
    if center_y != 0:
        dy = (y_local - center_y) / center_y
    else:
        dy = 0.0

    # Use pixel-scale skew (matches overlay)
    x_local = x_local + dx * skew_x * center_x
    y_local = y_local + dy * skew_y * center_y

    abs_x = origin_x + x_local
    abs_y = origin_y + y_local

    # Optional safety clamp: keep inside primary screen bounds
    sx_screen, sy_screen = get_screen_size()
    abs_x = max(0.0, min(abs_x, sx_screen - 1.0))
    abs_y = max(0.0, min(abs_y, sy_screen - 1.0))

    return abs_x, abs_y


# ---------------- Tempo tool automation ----------------

def apply_tempo_to_delays():
    """Use the in-game tempo tool (slot 6) to set delay times on all delay blocks.

    For each non-empty tick (active tick), we compute the total time since the
    previous active tick, quantize it to game resolution (0.01, min 0.05),
    group equal values and apply them using the captured tempo_tool_pos.
    """
    global tempo, active_ticks, tick_positions, tempo_tool_pos, stop_play

    if tempo <= 0:
        output_text.insert(tk.END, "Tempo <= 0: cannot compute delays for tempo tool.\\n")
        return
    if not active_ticks or tick_positions is None:
        output_text.insert(tk.END, "No active ticks or tick positions; tempo tool step skipped.\\n")
        return
    if tempo_tool_pos is None:
        output_text.insert(tk.END, "Tempo UI position not captured; tempo tool step skipped.\\n")
        return

    # Make sure tool 6 is definitely equipped before we start working with groups
    try:
        keyboard.press_and_release('6')
        # Base wait so the game has time to display the tempo UI
        try:
            wait_s = float(tempo_tool_wait_var.get())
        except Exception:
            wait_s = 2.0
        if wait_s < 0:
            wait_s = 0.0
        time.sleep(wait_s)
    except Exception:
        pass

    # Compute continuous delay (seconds) between each active tick and the previous one.
    # First active tick should still get at least one "tick" of delay so the
    # first delay block is not treated as 0.
    delays = []  # same length as active_ticks
    if not active_ticks:
        return

    for idx, tick in enumerate(active_ticks):
        if idx == 0:
            # First active tick: at least 1 tick worth of delay
            gap_ticks = max(1, tick - 0)
        else:
            prev_tick = active_ticks[idx - 1]
            gap_ticks = max(1, tick - prev_tick)
        delay_sec = gap_ticks / tempo
        delays.append(delay_sec)

    # Quantize each delay to game resolution and group by value
    groups = {}
    for idx, d in enumerate(delays):
        q = quantize_delay_to_step(d)
        if q <= 0.0:
            continue
        groups.setdefault(q, []).append(idx)

    if not groups:
        output_text.insert(tk.END, "No positive delays after quantization; nothing to apply.\\n")
        return

    output_text.insert(tk.END, f"Tempo tool: {len(groups)} delay groups to apply.\\n")

    for delay_value, indices in sorted(groups.items()):
        if stop_play:
            output_text.insert(tk.END, "Tempo tool aborted by user.\\n")
            return

        # Re-equip tool 6 twice to ensure selection is cleared and tool is active
        try:
            keyboard.press_and_release('6')
            time.sleep(0.03)
            keyboard.press_and_release('6')
            time.sleep(0.03)
        except Exception:
            pass

        # Multi-select the delay blocks for this group:
        # - first click without Shift to start selection
        # - remaining clicks with Shift held to extend selection
        if indices:
            sorted_indices = sorted(indices)
        else:
            sorted_indices = []

        if not sorted_indices:
            continue

        # First delay: normal click
        first_i = sorted_indices[0]
        try:
            x0, y0 = tick_positions[first_i]
        except Exception:
            x0 = y0 = None
        if x0 is not None:
            move_and_click_abs(x0, y0, pause_before_click=0.004)
            time.sleep(0.04)

        # Remaining delays: Shift-click to add to selection
        rest_indices = sorted_indices[1:]
        if rest_indices:
            try:
                keyboard.press('shift')
            except Exception:
                pass
            for i in rest_indices:
                if stop_play:
                    break
                try:
                    x, y = tick_positions[i]
                except Exception:
                    continue
                move_and_click_abs(x, y, pause_before_click=0.004)
                time.sleep(0.01)
            try:
                keyboard.release('shift')
            except Exception:
                pass

        if stop_play:
            output_text.insert(tk.END, "Tempo tool aborted by user during selection.\\n")
            return

        # Focus tempo UI, paste value, press Enter
        x_ui, y_ui = tempo_tool_pos
        # Give the UI a bit more time per group as well
        time.sleep(0.5)
        move_and_click_abs(x_ui, y_ui, pause_before_click=0.005)

        value_str = f"{delay_value:.2f}"
        try:
            root.clipboard_clear()
            root.clipboard_append(value_str)
        except Exception:
            pass
        try:
            keyboard.press_and_release('ctrl+v')
            time.sleep(0.03)
            keyboard.press_and_release('enter')
        except Exception:
            pass

        # Small pause, then press 6 twice to save and deselect
        time.sleep(0.05)
        try:
            keyboard.press_and_release('6')
            time.sleep(0.03)
            keyboard.press_and_release('6')
        except Exception:
            pass

        output_text.insert(tk.END, f"Tempo tool: applied {value_str} to {len(indices)} delays.\\n")


# ---------------- Play logic ----------------
def play_song_thread():
    global song_ticks, notes_by_tick, note_positions, tempo, stop_play, active_ticks
    if song_ticks <= 0:
        output_text.insert(tk.END, "No song loaded.\n"); return

    compute_required_counts()

    # If there are no active ticks, nothing to play/click
    if not active_ticks:
        output_text.insert(tk.END, "No non-empty ticks after compression. Nothing to build.\n")
        return

    stop_play = False
    countdown = 5.0
    start_time = time.time()
    while time.time() - start_time < countdown:
        if stop_play:
            output_text.insert(tk.END, "Playback cancelled.\n"); return
        time.sleep(0.05)

    prev_tick = 0
    # Iterate only over active (non-empty) ticks. We still keep correct timing
    # by using the tick difference (tick - prev_tick) / tempo.
    last_index = len(active_ticks) - 1
    for idx, tick in enumerate(active_ticks):
        if stop_play:
            output_text.insert(tk.END, "Playback stopped by user.\\n")
            break

        notes_in_tick = notes_by_tick.get(tick, [])  # should always be non-empty here
        delay_sec = (tick - prev_tick) / tempo if tempo > 0 else 0.0

        if notes_in_tick:
            needed_by_name = {}
            for n in notes_in_tick:
                name = key_to_note_name(n.key)
                needed_by_name[name] = needed_by_name.get(name, 0) + 1

            for name, needed in needed_by_name.items():
                while len(note_positions.get(name, [])) < needed:
                    output_text.insert(tk.END, f"Waiting for {needed - len(note_positions.get(name, []))} capture(s) for {name}... Press F6.\n")
                    waited = 0.0
                    while len(note_positions.get(name, [])) < needed and waited < 10.0:
                        if stop_play: break
                        time.sleep(0.1); waited += 0.1
                    if stop_play: break
                    if len(note_positions.get(name, [])) < needed:
                        output_text.insert(tk.END, f"Insufficient captures for {name}; will skip missing instances.\n")
                        break
                if stop_play: break
            if stop_play: break

            used_indices = {name: 0 for name in needed_by_name.keys()}
            for n in notes_in_tick:
                if stop_play: break
                name = key_to_note_name(n.key)
                idx = used_indices.get(name, 0)
                positions = note_positions.get(name, [])
                if idx < len(positions):
                    x, y = positions[idx]
                    used_indices[name] = idx + 1
                    move_and_click_abs(x, y)
                    # wait between individual note clicks according to user setting
                    try:
                        wait_ms = float(move_between_notes_var.get())
                    except Exception:
                        wait_ms = 6.0
                    time.sleep(max(0.0, wait_ms) / 1000.0)
                else:
                    output_text.insert(tk.END, f"Missing capture for {name} instance; skipping that note's click.\n")

            tick_pos = compute_screen_coords_for_tick(tick)
            if tick_pos:
                # if we had notes this tick, wait move_to_delay_ms before moving to delay cell
                if notes_in_tick:
                    try:
                        wait_ms = float(move_to_delay_var.get())
                    except Exception:
                        wait_ms = 20.0
                    time.sleep(max(0.0, wait_ms) / 1000.0)
                move_and_click_abs(tick_pos[0], tick_pos[1])
                # after clicking the delay cell, small pause
                time.sleep(0.004)

                # If this was the last active tick, equip tempo tool (slot 6)
                if idx == last_index:
                    try:
                        keyboard.press_and_release('6')
                    except Exception:
                        pass
            else:
                output_text.insert(tk.END, f"No tick cell mapping for tick {tick}; delay cell click skipped.\\n")
        else:
            if delay_sec > 0:
                waited = 0.0; step = 0.01
                while waited < delay_sec:
                    if stop_play: break
                    time.sleep(min(step, delay_sec - waited)); waited += min(step, delay_sec - waited)
        prev_tick = tick

    output_text.insert(tk.END, "Playback finished or stopped.\\n")

    # After building all delay blocks, automatically run the tempo tool
    # to set delay times based on computed tick timings.
    try:
        apply_tempo_to_delays()
    except Exception as e:
        try:
            output_text.insert(tk.END, f"Tempo tool error: {e}\\n")
        except Exception:
            pass

def play_song():
    global stop_play
    stop_play = False
    threading.Thread(target=play_song_thread, daemon=True).start()

# ---------------- GUI construction (compact) ----------------
root = tk.Tk()
root.title("NBS Overlay Music maker")
# adjusted default window: a bit narrower, a bit taller
root.geometry("800x560")

# Output area (smaller)
output_text = scrolledtext.ScrolledText(root, width=100, height=12)
output_text.pack(padx=6, pady=6)

# Scrollable note capture area (compact)
notes_container = tk.Frame(root)
notes_container.pack(fill="x", padx=6)

canvas_notes = tk.Canvas(notes_container, height=140)
scrollbar_notes = tk.Scrollbar(notes_container, orient="vertical", command=canvas_notes.yview)
canvas_notes.configure(yscrollcommand=scrollbar_notes.set)
scrollbar_notes.pack(side="right", fill="y")
canvas_notes.pack(side="left", fill="both", expand=True)

frame_notes = tk.Frame(canvas_notes)
canvas_notes.create_window((0,0), window=frame_notes, anchor='nw')

def _on_frame_config(event):
    canvas_notes.configure(scrollregion=canvas_notes.bbox("all"))
frame_notes.bind("<Configure>", _on_frame_config)

# allow mousewheel scrolling on note area
def _on_mousewheel_notes(event):
    canvas_notes.yview_scroll(int(-1*(event.delta/120)), "units")
canvas_notes.bind_all("<MouseWheel>", _on_mousewheel_notes)

note_buttons = {}
for i, note in enumerate(NOTE_NAMES):
    b = tk.Button(frame_notes, text=note, width=10, command=lambda n=note: capture_note_position(n))
    b.grid(row=i//6, column=i%6, padx=2, pady=2)
    note_buttons[note] = b

# Corner capture buttons in a compact row
corner_frame = tk.Frame(root)
corner_frame.pack(fill="x", padx=6, pady=(4,0))
corner_buttons = {}
for corner in ["top_left","bottom_right"]:
    btn = tk.Button(corner_frame, text=corner.replace("_"," ").title(), width=14, command=lambda c=corner: capture_grid_corner(c))
    btn.pack(side="left", padx=6)
    corner_buttons[corner] = btn


# Controls row (compact)
controls = tk.Frame(root)
controls.pack(pady=6, fill="x", padx=6)

load_btn = tk.Button(controls, text="Load NBS", width=10, command=load_nbs)
load_btn.pack(side="left", padx=4)

show_grid_btn = tk.Button(controls, text="Show Grid", width=10, command=toggle_overlay)
show_grid_btn.pack(side="left", padx=4)

# Reset Notes button (new)
def reset_note_captures():
    global note_positions
    note_positions = {note: [] for note in NOTE_NAMES}
    compute_required_counts()
    output_text.insert(tk.END, "All note captures reset.\n")

reset_notes_btn = tk.Button(controls, text="Reset Notes", width=10, command=reset_note_captures)
reset_notes_btn.pack(side="left", padx=4)

# grid settings compact
cols_entry = tk.Entry(controls, width=4)
rows_entry = tk.Entry(controls, width=4)

tk.Label(controls, text="Cols:").pack(side="left", padx=(8,0))
cols_entry.pack(side="left", padx=2)
cols_entry.bind("<Return>", lambda e: set_columns_from_entry())

tk.Label(controls, text="Rows:").pack(side="left", padx=(6,0))
rows_entry.pack(side="left", padx=2)
rows_entry.bind("<Return>", lambda e: set_rows_from_entry())

apply_grid_btn = tk.Button(controls, text="Apply Grid", width=10, command=apply_grid_settings)
apply_grid_btn.pack(side="left", padx=6)

# dot size + movement controls (compact)
dot_size_var = tk.IntVar(value=2)

# NEW: timing settings (ms)
move_between_notes_var = tk.DoubleVar(value=5.0)   # ms between note clicks
move_to_delay_var = tk.DoubleVar(value=400.0)      # ms before moving to delay cell

# Wait before tempo tool UI is used (seconds)
# This gives the game time to show the tool 6 window
tempo_tool_wait_var = tk.DoubleVar(value=2.0)

# place smaller control widgets
ctrl2 = tk.Frame(root)
ctrl2.pack(fill="x", padx=6)

tk.Label(ctrl2, text="Dot:").pack(side="left")
dot_size_spin = tk.Spinbox(ctrl2, from_=1, to=8, width=3, textvariable=dot_size_var)
dot_size_spin.pack(side="left", padx=4)

# timing controls
tk.Label(ctrl2, text="Between notes (ms):").pack(side="left", padx=(8,0))
move_between_spin = tk.Spinbox(ctrl2, from_=0.0, to=1000.0, increment=1.0, width=6, textvariable=move_between_notes_var)
move_between_spin.pack(side="left", padx=4)

tk.Label(ctrl2, text="To delay (ms):").pack(side="left", padx=(8,0))
move_to_delay_spin = tk.Spinbox(ctrl2, from_=0.0, to=2000.0, increment=1.0, width=6, textvariable=move_to_delay_var)
move_to_delay_spin.pack(side="left", padx=4)

# Tempo tool wait (seconds)
tk.Label(ctrl2, text="Tempo wait (s):").pack(side="left", padx=(8,0))
tempo_wait_spin = tk.Spinbox(ctrl2, from_=0.0, to=10.0, increment=0.5, width=5, textvariable=tempo_tool_wait_var)
tempo_wait_spin.pack(side="left", padx=4)

# Wiggle controls (new)
wiggle_amp_var = tk.DoubleVar(value=2.4)  # pixels
wiggle_count_var = tk.IntVar(value=12)

tk.Label(ctrl2, text="Wiggle px:").pack(side="left", padx=(8,0))
wiggle_amp_spin = tk.Spinbox(ctrl2, from_=0.0, to=20.0, increment=0.5, width=5, textvariable=wiggle_amp_var)
wiggle_amp_spin.pack(side="left", padx=4)

tk.Label(ctrl2, text="Wiggle cnt:").pack(side="left", padx=(6,0))
wiggle_count_spin = tk.Spinbox(ctrl2, from_=0, to=20, width=3, textvariable=wiggle_count_var)
wiggle_count_spin.pack(side="left", padx=4)

# Skew controls (small)
skew_frame = tk.Frame(root)
skew_frame.pack(fill="x", padx=6, pady=(6,4))

# Use DoubleVar to hold numeric values; entries display formatted 0.00 and buttons adjust by 0.01
skew_x_var = tk.DoubleVar(value=0.0)
skew_y_var = tk.DoubleVar(value=0.0)

tk.Label(skew_frame, text="Skew X").pack(side="left")
skew_x_entry = tk.Entry(skew_frame, width=8)
skew_x_entry.pack(side="left", padx=6)
skew_x_entry.insert(0, "0.00")

def apply_skew_x(e=None):
    try:
        v = float(skew_x_entry.get())
    except Exception:
        v = 0.0
    skew_x_var.set(v)
    skew_x_entry.delete(0, tk.END)
    skew_x_entry.insert(0, f"{v:.2f}")
    # mapping changed
    update_mapping()
    update_or_create_overlay()

skew_x_entry.bind("<Return>", apply_skew_x)

# +/- buttons for Skew X
def _inc_skew_x():
    v = round(skew_x_var.get() + 0.01, 2)
    skew_x_var.set(v)
    skew_x_entry.delete(0, tk.END); skew_x_entry.insert(0, f"{v:.2f}")
    update_mapping(); update_or_create_overlay()

def _dec_skew_x():
    v = round(skew_x_var.get() - 0.01, 2)
    skew_x_var.set(v)
    skew_x_entry.delete(0, tk.END); skew_x_entry.insert(0, f"{v:.2f}")
    update_mapping(); update_or_create_overlay()

tk.Button(skew_frame, text="+0.01", width=5, command=_inc_skew_x).pack(side="left", padx=2)
tk.Button(skew_frame, text="-0.01", width=5, command=_dec_skew_x).pack(side="left", padx=2)

tk.Label(skew_frame, text="Skew Y").pack(side="left", padx=(8,0))
skew_y_entry = tk.Entry(skew_frame, width=8)
skew_y_entry.pack(side="left", padx=6)
skew_y_entry.insert(0, "0.00")

def apply_skew_y(e=None):
    try:
        v = float(skew_y_entry.get())
    except Exception:
        v = 0.0
    skew_y_var.set(v)
    skew_y_entry.delete(0, tk.END)
    skew_y_entry.insert(0, f"{v:.2f}")
    update_mapping(); update_or_create_overlay()

skew_y_entry.bind("<Return>", apply_skew_y)

# +/- buttons for Skew Y
def _inc_skew_y():
    v = round(skew_y_var.get() + 0.01, 2)
    skew_y_var.set(v)
    skew_y_entry.delete(0, tk.END); skew_y_entry.insert(0, f"{v:.2f}")
    update_mapping(); update_or_create_overlay()

def _dec_skew_y():
    v = round(skew_y_var.get() - 0.01, 2)
    skew_y_var.set(v)
    skew_y_entry.delete(0, tk.END); skew_y_entry.insert(0, f"{v:.2f}")
    update_mapping(); update_or_create_overlay()

tk.Button(skew_frame, text="+0.01", width=5, command=_inc_skew_y).pack(side="left", padx=2)
tk.Button(skew_frame, text="-0.01", width=5, command=_dec_skew_y).pack(side="left", padx=2)

reset_skew_btn = tk.Button(skew_frame, text="Reset Skew", command=lambda: (skew_x_var.set(0.0), skew_y_var.set(0.0), skew_x_entry.delete(0, tk.END), skew_x_entry.insert(0, "0.00"), skew_y_entry.delete(0, tk.END), skew_y_entry.insert(0, "0.00"), update_mapping(), update_or_create_overlay()))
reset_skew_btn.pack(side="left", padx=6)

# Tempo tool controls: capture tempo UI position + rounding mode
tempo_frame = tk.Frame(root)
tempo_frame.pack(fill="x", padx=6, pady=(4,4))

tempo_tool_btn = tk.Button(tempo_frame, text="Capture Tempo UI", width=18)
tempo_tool_btn.pack(side="left", padx=4)

def capture_tempo_tool_pos():
    """Capture screen position of the in-game tempo input/apply UI (via F6)."""
    tempo_tool_btn.config(text="Tempo UI: Waiting...")

    def wait_f6():
        global tempo_tool_pos
        keyboard.wait("f6")
        x, y = get_cursor_pos()
        tempo_tool_pos = (x, y)
        root.after(0, lambda: tempo_tool_btn.config(text=f"Tempo UI: ({x},{y})"))

    threading.Thread(target=wait_f6, daemon=True).start()

tempo_tool_btn.config(command=capture_tempo_tool_pos)

# Round mode: down (faster) or up (longer) when snapping to game step (e.g. 0.05)
round_mode_var = tk.StringVar(value="down")

tk.Label(tempo_frame, text="Tempo round:").pack(side="left", padx=(8,0))
tk.Radiobutton(tempo_frame, text="Down (faster)", variable=round_mode_var, value="down").pack(side="left", padx=2)
tk.Radiobutton(tempo_frame, text="Up (longer)", variable=round_mode_var, value="up").pack(side="left", padx=2)

overlay_status_var = tk.StringVar(value="Overlay: hidden")
status_label = tk.Label(root, textvariable=overlay_status_var)
status_label.pack(side="left", padx=6)

play_button = tk.Button(root, text="Play Song", width=12, command=play_song)
play_button.pack(side="right", padx=8, pady=6)

# Always on top flag for main window and overlay
always_on_top_var = tk.BooleanVar(value=False)

def set_always_on_top():
    val = bool(always_on_top_var.get())
    try:
        root.attributes("-topmost", val)
    except Exception:
        pass
    try:
        if overlay_window is not None:
            overlay_window.attributes("-topmost", val)
    except Exception:
        pass

# CHECKBOX OBOK SKEW
always_chk = tk.Checkbutton(
    skew_frame,
    text="Always on top",
    variable=always_on_top_var,
    onvalue=True,
    offvalue=False,
    command=set_always_on_top
)
always_chk.pack(side="left", padx=(8,4))

# bezpieczne ustawienie przy starcie
try:
    if overlay_window is not None:
        overlay_window.attributes("-topmost", bool(always_on_top_var.get()))
except Exception:
    pass


def on_close():
    """Cleanly close the app and unhook global keyboard listeners."""
    try:
        keyboard.unhook_all()
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass

root.protocol("WM_DELETE_WINDOW", on_close)

# ---------------- Register keyboard handlers ----------------
# Register F6 handler for capturing note positions
keyboard.on_press_key("f6", _f6_global_handler)

# Register F7 handler for stopping playback (fixed)
def _on_f7(e=None):
    global stop_play
    stop_play = True
    try:
        root.after(0, lambda: output_text.insert(tk.END, "F7 pressed — stopping playback.\n"))
    except Exception:
        pass

keyboard.on_press_key("f7", _on_f7)

# ---------------- Start periodic refresh and mainloop ----------------
def periodic_refresh():
    # update overlay if enabled (but only redraw when needed)
    try:
        if overlay_shown and overlay_window is not None:
            update_overlay()
    except Exception:
        pass

    # update note button labels (armed / counts) - lightweight
    try:
        with armed_lock:
            armed = armed_note
        for note, btn in note_buttons.items():
            have = len(note_positions.get(note, []))
            need = required_counts.get(note, 0)
            if note == armed:
                # show Armed label if this note is armed
                if need > 0:
                    btn.config(text=f"{note}: Armed {have}/{need}")
                else:
                    btn.config(text=f"{note}: Armed {have}")
            else:
                if need > 0:
                    btn.config(text=f"{note}: {have}/{need}")
                else:
                    btn.config(text=f"{note}: {have}")
    except Exception:
        pass

    # schedule next refresh
    root.after(200, periodic_refresh)

if __name__ == "__main__":
    # Start the periodic refresh loop
    root.after(200, periodic_refresh)

    # Start the GUI main loop
    root.mainloop()
