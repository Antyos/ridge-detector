import json
import threading
import tkinter as tk
from collections.abc import Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TypeGuard

import numpy as np
from PIL import Image, ImageTk

from ridge_detector.detector import RidgeData, RidgeDetector, RidgeDetectorConfig


class InvalidLineWidthError(ValueError): ...


def is_int_list(lst: Sequence[int | None]) -> TypeGuard[list[int]]:
    """Check if all elements in the list are integers."""
    return all(isinstance(x, int) for x in lst)


def parse_line_widths(line_widths: str) -> list[int]:
    """Convert line widths from string to list of integers."""
    if not line_widths:
        raise InvalidLineWidthError("No line widths provided")
    widths = []
    for w in line_widths.split(","):
        # Support slice syntax
        if ":" in w:
            segments = [seg.strip() for seg in w.split(":")]
            if len(segments) > 3:
                raise InvalidLineWidthError(
                    f"Too many segments in line width range: {len(segments)}"
                )
            try:
                int_segments = [int(seg) if seg else None for seg in segments]
            except ValueError:
                raise InvalidLineWidthError(
                    f"Non integer in line width range: {segments}"
                )
            if int_segments[0] is None:
                int_segments[0] = 1
            if not is_int_list(int_segments):
                raise InvalidLineWidthError(f"Ambiguous line width range: {segments}")
            if any(seg <= 0 for seg in int_segments):
                raise InvalidLineWidthError(
                    f"Cannot use negative or zero values: {segments}"
                )
            # Use inclusive stop condition
            if len(int_segments) > 1:
                int_segments[1] += 1
            widths.extend(range(*int_segments))
        # Single digit
        elif w.strip().isdigit():
            try:
                value = int(w.strip())
            except ValueError:
                raise InvalidLineWidthError(f"Invalid line width: {w}")
            if value <= 0:
                raise InvalidLineWidthError(
                    f"Cannot use negative or zero values: {value}"
                )
            widths.append(value)
        else:
            raise InvalidLineWidthError(f"Invalid line width: {w}")
    return widths


class RidgeDetectorGUI(tk.Tk):
    img: Image.Image | None = None
    ridge_image: Image.Image | None = None
    ridge_data: RidgeData | None = None
    image_path: Path | None = None

    def __init__(self):
        super().__init__()
        self.title("Ridge Detector GUI")
        self.geometry("900x600")

        # Main frame
        self.paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        # Display the sash. See: https://stackoverflow.com/a/79506039
        ttk.Style().configure("TPanedwindow", background="dark grey")
        ttk.Style().configure("Sash", sashthickness=2)

        # Left: Image display
        self.image_frame = ttk.Frame(self.paned_window)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.image_frame, bg="gray", width=600, height=600)
        self.canvas.pack(fill=tk.BOTH)
        self.paned_window.add(self.image_frame, weight=3)

        # For canvas-based zoom and pan
        self.canvas.bind("<MouseWheel>", self.on_zoom)  # Windows
        self.canvas.bind("<Button-4>", self.on_zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)  # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_move)

        # Right: Parameters section (empty for now)
        self.param_frame = ttk.Frame(self.paned_window, width=300)
        # self.param_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.paned_window.add(self.param_frame, weight=1)

        param_label = ttk.Label(
            self.param_frame,
            text="Ridge Detector Parameters",
            font=("Arial", 14, "bold"),
        )
        param_label.pack(anchor=tk.W, padx=10, pady=10)

        # RidgeDetectorConfig parameter fields
        # line_widths (as comma-separated string)
        self.line_widths_var = tk.StringVar(value="1,2")
        self.line_widths_var.trace_add("write", self.on_params_update)
        ttk.Label(
            self.param_frame,
            text="Line Widths (comma-separated or slice notation)",
        ).pack(anchor=tk.W, padx=10)
        ttk.Entry(
            self.param_frame,
            textvariable=self.line_widths_var,
        ).pack(fill=tk.X, padx=10, pady=2)

        # low_contrast
        self.low_contrast_var = tk.IntVar(value=100)
        self.low_contrast_var.trace_add("write", self.on_params_update)
        self.high_contrast_var = tk.IntVar(value=200)
        self.high_contrast_var.trace_add("write", self.on_params_update)
        ttk.Label(self.param_frame, text="Low Contrast").pack(anchor=tk.W, padx=10)
        low_contrast_frame = ttk.Frame(self.param_frame)
        low_contrast_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Scale(
            low_contrast_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.low_contrast_var,
            command=lambda value: self.low_contrast_var.set(
                min(int(float(value)), self.high_contrast_var.get())
            ),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Spinbox(
            low_contrast_frame,
            from_=0,
            to=255,
            textvariable=self.low_contrast_var,
            width=5,
            increment=1,
        ).pack(fill=tk.X, padx=10, pady=2)

        # high_contrast
        ttk.Label(self.param_frame, text="High Contrast").pack(anchor=tk.W, padx=10)
        high_contrast_frame = ttk.Frame(self.param_frame)
        high_contrast_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Scale(
            high_contrast_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.high_contrast_var,
            command=lambda value: self.high_contrast_var.set(
                max(int(float(value)), self.low_contrast_var.get())
            ),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Spinbox(
            high_contrast_frame,
            from_=0,
            to=255,
            textvariable=self.high_contrast_var,
            width=5,
            increment=1,
        ).pack(fill=tk.X, padx=10, pady=2)

        # min_len
        self.min_len_var = tk.IntVar(value=5)
        self.min_len_var.trace_add("write", self.on_params_update)
        ttk.Label(self.param_frame, text="Min Length").pack(anchor=tk.W, padx=10)
        ttk.Spinbox(
            self.param_frame,
            from_=0,
            to=float("inf"),
            textvariable=self.min_len_var,
            increment=1,
        ).pack(fill=tk.X, padx=10, pady=2)

        # max_len
        self.max_len_var = tk.IntVar(value=0)
        self.max_len_var.trace_add("write", self.on_params_update)
        ttk.Label(self.param_frame, text="Max Length (0 = no max)").pack(
            anchor=tk.W, padx=10
        )
        ttk.Spinbox(
            self.param_frame,
            from_=0,
            to=float("inf"),
            textvariable=self.max_len_var,
            increment=1,
        ).pack(fill=tk.X, padx=10, pady=2)

        # dark_line
        self.dark_line_var = tk.BooleanVar(value=True)
        self.dark_line_var.trace_add("write", self.on_params_update)
        ttk.Checkbutton(
            self.param_frame, text="Detect Dark Lines", variable=self.dark_line_var
        ).pack(anchor=tk.W, padx=10, pady=2)

        # estimate_width
        self.estimate_width_var = tk.BooleanVar(value=True)
        self.estimate_width_var.trace_add("write", self.on_params_update)
        ttk.Checkbutton(
            self.param_frame,
            text="Estimate Width",
            variable=self.estimate_width_var,
        ).pack(anchor=tk.W, padx=10, pady=2)

        # extend_line
        self.extend_line_var = tk.BooleanVar(value=False)
        self.extend_line_var.trace_add("write", self.on_params_update)
        ttk.Checkbutton(
            self.param_frame, text="Extend Line", variable=self.extend_line_var
        ).pack(anchor=tk.W, padx=10, pady=2)

        # correct_pos
        self.correct_pos_var = tk.BooleanVar(value=False)
        self.correct_pos_var.trace_add("write", self.on_params_update)
        ttk.Checkbutton(
            self.param_frame, text="Correct Position", variable=self.correct_pos_var
        ).pack(anchor=tk.W, padx=10, pady=2)

        ttk.Button(self.param_frame, text="Apply", command=self.display_ridges).pack(
            anchor=tk.W, padx=10, pady=2
        )

        # Menu
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(
            label="Open Image", command=self.open_image, accelerator="(Ctrl+O)"
        )
        filemenu.add_command(
            label="Save Ridge Image",
            command=self.save_ridge_image,
            accelerator="(Ctrl+S)",
        )
        filemenu.add_command(label="Export Ridge Data", command=self.export_ridges)
        filemenu.add_separator()
        filemenu.add_command(label="Export Parameters", command=self.export_params)
        filemenu.add_command(label="Import Parameters", command=self.import_params)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(
            label="Zoom to Fit", command=self.zoom_to_fit_image, accelerator="(F)"
        )
        menubar.add_cascade(label="View", menu=viewmenu)
        self.config(menu=menubar)

        # Open image
        self.bind_all("<Control-o>", lambda event: self.open_image())
        self.bind_all("<Control-s>", lambda event: self.save_ridge_image())
        self.bind_all("<f>", lambda event: self.zoom_to_fit_image())

        self.img = None
        self.ridge_image = None
        self.img_scale = 1.0
        self.tk_img = None
        self.img_canvas_id = None
        self._detector_thread = None
        self._ridge_detector_lock = threading.Lock()
        self._ridge_detector_pending = False
        # Need to call update() before getting the image_frame dimensions
        self.update()
        self.img_x = self.image_frame.winfo_width() // 2
        self.img_y = self.image_frame.winfo_height() // 2

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ]
        )
        if not file_path:
            return
        img = Image.open(file_path)
        self.image_path = Path(file_path)
        self.img = img
        self.ridge_image = None
        self.ridge_data = None
        self.zoom_to_fit_image()  # Also calls display_image()
        self.display_ridges()

    def display_image(self):
        # NOTE: Using PIL for scalling is not very performant for large images,
        # because pixels outside the viewbox aren't culled.
        if self.img is None:
            return
        image = self.ridge_image if self.ridge_image is not None else self.img
        scaled_image = image.resize(
            (
                int(image.width * self.img_scale),
                int(image.height * self.img_scale),
            ),
            Image.Resampling.NEAREST,
        )
        self.tk_img = ImageTk.PhotoImage(scaled_image)
        self.canvas.delete("all")
        self.img_canvas_id = self.canvas.create_image(
            self.img_x, self.img_y, anchor=tk.CENTER, image=self.tk_img
        )

    def zoom_to_fit_image(self):
        if self.img is None:
            return
        self.update()
        width = self.image_frame.winfo_width()
        height = self.image_frame.winfo_height()
        self.img_x = width // 2
        self.img_y = height // 2
        self.img_scale = min(width / self.img.width, height / self.img.height)
        self.display_image()

    def on_params_update(self, name: str, index: str, mode: str):
        self.display_ridges()

    def get_line_widths(self) -> list[int]:
        line_widths = self.line_widths_var.get()
        return parse_line_widths(line_widths)

    def get_ridge_detector_params(self) -> RidgeDetectorConfig:
        return RidgeDetectorConfig(
            line_widths=self.get_line_widths(),
            low_contrast=self.low_contrast_var.get(),
            high_contrast=self.high_contrast_var.get(),
            min_len=self.min_len_var.get(),
            max_len=self.max_len_var.get(),
            dark_line=self.dark_line_var.get(),
            estimate_width=self.estimate_width_var.get(),
            extend_line=self.extend_line_var.get(),
            correct_pos=self.correct_pos_var.get(),
        )

    def display_ridges(self):
        if self.img is None:
            return
        try:
            params = self.get_ridge_detector_params()
        except (tk.TclError, InvalidLineWidthError):
            # Invalid values in parameter input fields
            return
        with self._ridge_detector_lock:
            # Queue up at most one more ridge detection
            if self._detector_thread is not None and self._detector_thread.is_alive():
                self._ridge_detector_pending = True
                return
            self._detector_thread = threading.Thread(
                target=self.detect_lines, args=(params, self.img), daemon=True
            )
            self._detector_thread.start()

    def detect_lines(self, params: RidgeDetectorConfig, img: Image.Image):
        detector = RidgeDetector(params)
        self.ridge_data = detector.detect_lines(np.array(img))
        self.ridge_image = Image.fromarray(
            self.ridge_data.get_image_contours(params.estimate_width)
        )
        self.display_image()
        # If detection was pending, re-run it
        if self._ridge_detector_pending:
            self._ridge_detector_pending = False
            self.detect_lines(params, img)

    def save_ridge_image(self):
        if self.ridge_image is None:
            messagebox.showinfo("Error", "No ridge image to export.")
            return
        if self.image_path:
            initial_file = f"{self.image_path.stem}.ridges.png"
        else:
            initial_file = "ridge_image.png"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=initial_file,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("TIFF files", "*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.ridge_image.save(file_path)

    def export_ridges(self):
        if self.ridge_data is None:
            messagebox.showinfo("Error", "No ridge data to export.")
            return
        if self.image_path:
            initial_file = f"{self.image_path.name}.ridges.csv"
        else:
            initial_file = "ridges.csv"
        file_path = filedialog.asksaveasfilename(
            title="Export Ridge Data",
            defaultextension=".csv",
            initialfile=initial_file,
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        file_path = Path(file_path)
        df = self.ridge_data.to_dataframe()
        try:
            if file_path.suffix == ".csv":
                df.to_csv(file_path, index=False)
            elif file_path.suffix == ".xlsx":
                df.to_excel(file_path, index=False)
            elif file_path.suffix == ".parquet":
                df.to_parquet(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format.")
        except (ImportError, ModuleNotFoundError) as e:
            messagebox.showerror(
                "Error", f"Failed to find engine to export as {file_path.suffix}: {e}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def export_params(self):
        try:
            params = {
                "line_widths": self.line_widths_var.get(),
                "low_contrast": self.low_contrast_var.get(),
                "high_contrast": self.high_contrast_var.get(),
                "min_len": self.min_len_var.get(),
                "max_len": self.max_len_var.get(),
                "dark_line": self.dark_line_var.get(),
                "estimate_width": self.estimate_width_var.get(),
                "extend_line": self.extend_line_var.get(),
                "correct_pos": self.correct_pos_var.get(),
            }
        except tk.TclError:
            messagebox.showerror("Error", "Invalid parameters.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Export Parameters",
            defaultextension=".json",
            initialfile="parameters.json",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            with open(file_path, "w") as f:
                json.dump(params, f)

    def import_params(self):
        file_path = filedialog.askopenfilename(
            title="Import Parameters",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                params = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load parameters: {e}")
            return
        if "line_widths" in params:
            self.line_widths_var.set(params["line_widths"])
        if "low_contrast" in params:
            self.low_contrast_var.set(params["low_contrast"])
        if "high_contrast" in params:
            self.high_contrast_var.set(params["high_contrast"])
        if "min_len" in params:
            self.min_len_var.set(params["min_len"])
        if "max_len" in params:
            self.max_len_var.set(params["max_len"])
        if "dark_line" in params:
            self.dark_line_var.set(params["dark_line"])
        if "estimate_width" in params:
            self.estimate_width_var.set(params["estimate_width"])
        if "extend_line" in params:
            self.extend_line_var.set(params["extend_line"])
        if "correct_pos" in params:
            self.correct_pos_var.set(params["correct_pos"])

    def on_zoom(self, event: tk.Event):
        if self.img is None:
            return
        # Determine img scale factor
        if hasattr(event, "delta") and event.delta:
            self.img_scale += float(np.sign(event.delta) * 0.1)
        elif hasattr(event, "num"):
            self.img_scale += 0.1 if event.num == 4 else -0.1
        self.display_image()

    def on_pan_start(self, event: tk.Event):
        self._pan_x_start = event.x
        self._pan_y_start = event.y

    def on_pan_move(self, event: tk.Event):
        if self.img_canvas_id is None:
            return
        dx = event.x - self._pan_x_start
        dy = event.y - self._pan_y_start
        self.img_x += dx
        self.img_y += dy
        self.canvas.move(self.img_canvas_id, dx, dy)
        self._pan_x_start = event.x
        self._pan_y_start = event.y


if __name__ == "__main__":
    app = RidgeDetectorGUI()
    app.mainloop()
