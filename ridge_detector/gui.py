import json
import threading
import tkinter as tk
from collections.abc import Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
from typing import Any, NamedTuple, TypeGuard

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk
from tifffile import tifffile

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


class Size(NamedTuple):
    width: int
    height: int


class ImageData:
    path: Path
    handle: Image.Image | tifffile.TiffFile
    _image: Image.Image | NDArray
    width: int
    height: int
    dim_names: dict[str, str] = {}
    stack_shape: dict[str, int] = {}

    def __init__(self, file_path: Path):
        self.path = file_path
        if file_path.suffix in [".tif", ".tiff"]:
            self.handle = tifffile.TiffFile(file_path)
            self._image = self.handle.pages.first.asarray()
            if self.handle.is_imagej:
                img_data = self.handle.series[0].levels[0]
                self.dim_names = dict(zip(img_data.dims, img_data.axes))
                self.stack_shape = {
                    dim: size
                    for dim, size in img_data.sizes.items()
                    if dim not in {"width", "height", "sample"}
                }
                self.width = img_data.sizes["width"]
                self.height = img_data.sizes["height"]
            else:
                self.width = self.handle.pages.first.sizes["width"]
                self.height = self.handle.pages.first.sizes["height"]
        else:
            self.handle = Image.open(file_path)
            self._image = self.handle
            self.width = self.handle.width
            self.height = self.handle.height

    @property
    def is_stack(self):
        return len(self.stack_shape) > 0

    def _get_frame(self, **index: int):
        if isinstance(self.handle, tifffile.TiffFile):
            if self.handle.is_imagej:
                image_shape = self.handle.series[0].levels[0].sizes
                _num_z = image_shape.get("depth", 1)
                _num_c = image_shape.get("channel", 1)

                flat_index = (
                    (index.get("time", 0) * _num_z * _num_c)
                    + (index.get("depth", 0) * _num_c)
                    + index.get("channel", 0)
                )
                return self.handle.asarray(key=flat_index)
            return self.handle.pages.first.asarray()
        return self.handle

    def set_frame(self, **index: int):
        self._image = self._get_frame(**index)

    @property
    def image(self):
        return self._image

    @property
    def pil_image(self) -> Image.Image:
        if isinstance(self._image, np.ndarray):
            return Image.fromarray(self._image)
        return self._image

    @property
    def arr_image(self):
        if isinstance(self._image, np.ndarray):
            return self._image
        return np.array(self._image)


class RidgeDetectorGUI(tk.Tk):
    ridge_image: Image.Image | None = None
    ridge_data: RidgeData | None = None
    image_path: int
    _skip_ridge_display: bool = False

    def __init__(self):
        super().__init__()
        self._base_title = "Ridge Detector GUI"
        self.title(self._base_title)
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

        self.canvas = tk.Canvas(self.image_frame, bg="gray", width=600)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.paned_window.add(self.image_frame, weight=3)

        # For canvas-based zoom and pan
        self.canvas.bind("<MouseWheel>", self.on_zoom)  # Windows
        self.canvas.bind("<Button-4>", self.on_zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)  # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)  # Middle mouse
        self.canvas.bind("<B2-Motion>", self.on_pan_move)

        # Right: Parameters section (empty for now)
        self.param_frame = ttk.Frame(self.paned_window, width=300)
        # self.param_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.paned_window.add(self.param_frame, weight=1)

        param_label = ttk.Label(
            self.param_frame,
            text="Ridge Detector Parameters",
            font=("Arial", 14, "bold"),
        )
        param_label.pack(anchor=tk.W, padx=10, pady=2)

        copy_paste_frame = ttk.Frame(self.param_frame)
        copy_paste_frame.pack(fill=tk.X, padx=10, pady=2)
        self.copy_button = ttk.Button(
            copy_paste_frame, text="Copy", command=self.copy_parameters
        )
        self.copy_button.pack(side=tk.LEFT)
        self.paste_button = ttk.Button(
            copy_paste_frame, text="Paste", command=self.paste_parameters
        )
        self.paste_button.pack(side=tk.LEFT)

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
        ).pack(fill=tk.X, padx=10, pady=(2, 5))

        # low_contrast
        self.low_contrast_var = tk.IntVar(value=100)
        self.low_contrast_var.trace_add("write", self.on_params_update)
        self.high_contrast_var = tk.IntVar(value=200)
        self.high_contrast_var.trace_add("write", self.on_params_update)
        ttk.Label(self.param_frame, text="Low Contrast").pack(anchor=tk.W, padx=10)
        low_contrast_frame = ttk.Frame(self.param_frame)
        low_contrast_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
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
        ).pack(fill=tk.X, padx=10)

        # high_contrast
        ttk.Label(self.param_frame, text="High Contrast").pack(anchor=tk.W, padx=10)
        high_contrast_frame = ttk.Frame(self.param_frame)
        high_contrast_frame.pack(fill=tk.X, padx=10, pady=(2, 5))
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
        ).pack(fill=tk.X, padx=10)

        # min_len
        self.min_len_var = tk.IntVar(value=5)
        self.min_len_var.trace_add("write", self.on_params_update)
        ttk.Label(self.param_frame, text="Min Length (0=disable)").pack(
            anchor=tk.W, padx=10
        )
        ttk.Spinbox(
            self.param_frame,
            from_=0,
            to=float("inf"),
            textvariable=self.min_len_var,
            increment=1,
        ).pack(fill=tk.X, padx=10, pady=(2, 5))

        # max_len
        self.max_len_var = tk.IntVar(value=0)
        self.max_len_var.trace_add("write", self.on_params_update)
        ttk.Label(self.param_frame, text="Max Length (0=disable)").pack(
            anchor=tk.W, padx=10
        )
        ttk.Spinbox(
            self.param_frame,
            from_=0,
            to=float("inf"),
            textvariable=self.max_len_var,
            increment=1,
        ).pack(fill=tk.X, padx=10, pady=(2, 5))

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

        # Calculate row
        ttk.Separator(self.param_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=10, pady=5
        )
        self.auto_calculate_ridges = tk.BooleanVar(value=False)
        auto_calculate_row = ttk.Frame(self.param_frame)
        auto_calculate_row.pack(fill=tk.X, padx=10, pady=(2, 5))
        ttk.Button(
            auto_calculate_row,
            text="Calculate",
            command=lambda: self.update_ridges(force=True),
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            auto_calculate_row,
            text="Auto Calculate",
            variable=self.auto_calculate_ridges,
            # Run update_ridges() when toggled on
            command=lambda: self.auto_calculate_ridges.get() and self.update_ridges(),
        ).pack(side=tk.LEFT, padx=10)

        self.min_auto_line_width = tk.IntVar(value=1)
        ttk.Label(
            self.param_frame,
            text="Minimum line width for auto calculate",
        ).pack(anchor=tk.W, padx=10)
        ttk.Spinbox(
            self.param_frame,
            from_=0,
            to=float("inf"),
            textvariable=self.min_auto_line_width,
            increment=1,
        ).pack(fill=tk.X, padx=10)
        min_auto_line_width_description = ttk.Label(
            self.param_frame,
            text="Skip auto calculate when any line widths are below this value to reduce unnecessary computation. (0=disable)",
            font=(
                tkfont.nametofont("TkDefaultFont").actual()["family"],
                tkfont.nametofont("TkDefaultFont").actual()["size"],
                "italic",
            ),
            wraplength=1,
            justify=tk.LEFT,
        )
        min_auto_line_width_description.pack(anchor=tk.W, fill=tk.X, padx=10, pady=2)
        min_auto_line_width_description.bind(
            "<Configure>",
            lambda event, widget=min_auto_line_width_description: widget.config(
                wraplength=event.width - 25
            ),
        )

        # Status bar
        status_bar_frame = ttk.Frame(self.param_frame, relief="solid", borderwidth=1)
        status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar = ttk.Label(status_bar_frame, text="Status: Ready")
        self.status_bar.pack(side=tk.LEFT, padx=5)

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
        filemenu.add_command(label="Copy Ridge Data", command=self.copy_ridge_data)
        filemenu.add_separator()
        filemenu.add_command(label="Export Parameters", command=self.export_parameters)
        filemenu.add_command(label="Import Parameters", command=self.import_parameters)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(
            label="Zoom to Fit", command=self.zoom_to_fit, accelerator="(F)"
        )
        viewmenu.add_separator()
        # Ridge overlay
        self.overlay_mode = tk.StringVar(value="ridges")
        self.overlay_mode.trace_add(
            "write", lambda *_: (self._set_ridge_image(), self.display_image())
        )
        viewmenu.add_radiobutton(
            label="Show Image Only", variable=self.overlay_mode, value="base"
        )
        viewmenu.add_radiobutton(
            label="Show Ridges", variable=self.overlay_mode, value="ridges"
        )
        viewmenu.add_radiobutton(
            label="Show Binary Contours",
            variable=self.overlay_mode,
            value="binary_contours",
        )
        viewmenu.add_radiobutton(
            label="Show Binary Widths",
            variable=self.overlay_mode,
            value="binary_widths",
        )
        menubar.add_cascade(label="View", menu=viewmenu)
        self.config(menu=menubar)

        # Open image
        self.bind_all("<Control-o>", lambda event: self.open_image())
        self.bind_all("<Control-s>", lambda event: self.save_ridge_image())
        self.bind_all("<f>", lambda event: self.zoom_to_fit())

        self.image = None
        self.ridge_image = None
        self.tk_img = None
        self.img_canvas_id = None
        self.image_sliders = None
        self._detector_thread = None
        self._ridge_detector_lock = threading.Lock()
        self._ridge_detector_pending = None
        # Need to call update() before getting the image_frame dimensions
        self.update()
        self.img_x = self.image_frame.winfo_width() // 2
        self.img_y = self.image_frame.winfo_height() // 2
        self.img_scale = 1.0

        # Register close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def set_status(self, message: str):
        self.status_bar.config(text=message)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ]
        )
        if not file_path:
            return
        file_path = Path(file_path)
        self.set_status(f"Opening: '{file_path}'")
        self.image = ImageData(file_path)
        self.populate_image_sliders(self.image.stack_shape, self.image.dim_names)
        self.ridge_image = None
        self.ridge_data = None
        self.title(f"{file_path.name} - {self._base_title}")
        self.zoom_to_fit()  # Also calls display_image()
        self.update_ridges()
        self.set_status("Status: Ready")

    def populate_image_sliders(
        self, shape: dict[str, int], names: dict[str, str] | None = None
    ):
        if self.image_sliders is not None:
            self.image_sliders.destroy()
        if not shape:
            self.image_sliders = None
            return
        self.image_sliders = ttk.Frame(self.image_frame)
        self.image_sliders.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        self.image_index = {k: tk.IntVar(value=0) for k in shape.keys()}
        for axis, value in shape.items():
            int_var = self.image_index[axis]
            frame = ttk.Frame(self.image_sliders)
            frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
            if names is not None and axis in names:
                name = names[axis]
            else:
                name = axis
            ttk.Label(frame, text=f"{name}:").pack(side=tk.LEFT)
            slider = ttk.Scale(
                frame,
                from_=0,
                to=value - 1,
                orient=tk.HORIZONTAL,
                command=lambda val, var=int_var: var.set(int(float(val))),
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            ttk.Spinbox(
                frame, from_=0, to=value - 1, textvariable=int_var, width=5, increment=1
            ).pack(side=tk.LEFT)
            int_var.trace_add(
                "write",
                lambda *_: (
                    slider.set(int_var.get()),
                    self.on_slider_update(),
                ),
            )
            int_var.set(0)

    def on_slider_update(self):
        if self.image is None:
            return
        self.ridge_image = None
        index = {ax: var.get() for ax, var in self.image_index.items()}
        self.image.set_frame(**index)
        self.display_image()
        if self._detector_thread is not None:
            self._skip_ridge_display = True
        self.update_ridges()

    def display_image(self):
        # NOTE: Using PIL for scalling is not very performant for large images,
        # because pixels outside the viewbox aren't culled.
        if self.image is None:
            return
        if self.ridge_image is not None:
            image = self.ridge_image
        else:
            image = self.image.pil_image
        scaled_image = image.resize(
            (
                # Prevent scaling the image to size of 0
                max(int(image.width * self.img_scale), 1),
                max(int(image.height * self.img_scale), 1),
            ),
            Image.Resampling.NEAREST,
        )
        self.tk_img = ImageTk.PhotoImage(scaled_image)
        self.canvas.delete("all")
        self.img_canvas_id = self.canvas.create_image(
            self.img_x, self.img_y, anchor=tk.CENTER, image=self.tk_img
        )

    def zoom_to_fit(self):
        if self.image is None:
            return
        self.update()
        width = self.image_frame.winfo_width()
        height = self.image_frame.winfo_height()
        self.img_x = width // 2
        self.img_y = height // 2
        self.img_scale = min(width / self.image.width, height / self.image.height)
        self.display_image()

    def on_params_update(self, name: str, index: str, mode: str):
        self.update_ridges(force=False)

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

    def update_ridges(self, force: bool = False):
        auto_calculate = self.auto_calculate_ridges.get()
        if self.image is None or (not auto_calculate and not force):
            return
        try:
            params = self.get_ridge_detector_params()
        except (tk.TclError, InvalidLineWidthError):
            # Invalid values in parameter input fields
            return
        if (
            not force
            and auto_calculate
            and self.min_auto_line_width.get() > 0
            and any(np.asarray(params.line_widths) <= self.min_auto_line_width.get())
        ):
            self.set_status("WARNING: Low line widths detected")
            return
        with self._ridge_detector_lock:
            # Queue up at most one more ridge detection
            if self._detector_thread is not None and self._detector_thread.is_alive():
                self._ridge_detector_pending = (params, self.image.arr_image)
                return
            self._detector_thread = threading.Thread(
                target=self._detect_lines,
                args=(params, self.image.arr_image),
                daemon=True,
            )
            self._detector_thread.start()

    def _detect_lines(self, params: RidgeDetectorConfig, img_arr: NDArray):
        detector = RidgeDetector(params)
        self.set_status("Detecting ridges...")
        self.ridge_data = detector.detect_lines(img_arr)
        self._set_ridge_image(show_width=params.estimate_width)
        # Avoid showing old ridges after the frame has changed
        if self._skip_ridge_display:
            self._skip_ridge_display = False
        else:
            self.display_image()
        # If detection was pending, re-run it
        if self._ridge_detector_pending:
            next_params, next_img = self._ridge_detector_pending
            self._ridge_detector_pending = None
            self._detect_lines(next_params, next_img)
        self.set_status("Ready!")

    def _set_ridge_image(self, show_width: bool | None = None):
        if self.ridge_data is None:
            return
        if show_width is None:
            show_width = self.estimate_width_var.get()
        overlay_mode = self.overlay_mode.get()
        if overlay_mode == "base":
            ridge_image = None
        elif overlay_mode == "ridges":
            ridge_image = self.ridge_data.get_image_contours(show_width=show_width)
        elif overlay_mode == "binary_contours":
            ridge_image = self.ridge_data.get_binary_contours()
        elif overlay_mode == "binary_widths":
            ridge_image = self.ridge_data.get_binary_widths()
        else:
            raise ValueError(f"Invalid overlay mode: {overlay_mode}")
        # Set ridge image
        if ridge_image is not None:
            self.ridge_image = Image.fromarray(ridge_image)
        else:
            self.ridge_image = None

    def save_ridge_image(self):
        if self.ridge_image is None:
            messagebox.showinfo("Error", "No ridge image to export.")
            return
        if self.image:
            initial_file = f"{self.image.path.stem}.ridges.png"
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
            self.set_status(f"Saved ridge image to '{file_path}'")

    def export_ridges(self):
        if self.ridge_data is None:
            messagebox.showinfo("Error", "No ridge data to export.")
            return
        if self.image:
            initial_file = f"{self.image.path.name}.ridges.csv"
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
                return
            self.set_status(f"Exported ridge data to '{file_path}'")
        except (ImportError, ModuleNotFoundError) as e:
            messagebox.showerror(
                "Error", f"Failed to find engine to export as {file_path.suffix}: {e}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def copy_ridge_data(self):
        if self.ridge_data is None:
            messagebox.showinfo("Error", "No ridge data to copy.")
            return
        try:
            df = self.ridge_data.to_dataframe()
        except ImportError:
            messagebox.showerror(
                "Error", "Must have Pandas installed to copy as a dataframe"
            )
            return
        df.to_clipboard(index=False)
        self.set_status("Copied ridge data to clipboard")

    def dump_parameters(self) -> dict[str, Any] | None:
        try:
            return {
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
            return None

    def export_parameters(self):
        params = self.dump_parameters()
        if not params:
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
            self.set_status(f"Exported parameters to '{file_path}'")

    def copy_parameters(self):
        params = self.dump_parameters()
        if not params:
            messagebox.showerror("Error", "Invalid parameters.")
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(json.dumps(params))
            self.copy_button.config(text="Copied!")
            self.after(2000, lambda: self.copy_button.config(text="Copy"))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy parameters: {e}")

    def import_parameters(self):
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
        self.load_parameters(params)
        self.set_status("Imported parameters successfully.")

    def load_parameters(self, params: dict[str, Any]):
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

    def paste_parameters(self):
        try:
            params = json.loads(self.clipboard_get())
            self.load_parameters(params)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load parameters: {e}")

    def on_zoom(self, event: tk.Event):
        if self.image is None:
            return
        # Determine img scale factor
        if hasattr(event, "delta") and event.delta:
            delta = 1.25 if event.delta > 0 else 0.8
        elif hasattr(event, "num"):  # Linux scrolling
            delta = 1.25 if event.num == 4 else 0.8
        else:
            delta = 1
        # Minimum zoom level is 0.001
        self.img_scale = max(0.001, self.img_scale * delta)
        # Update offset so that we are scaling based on the center of the screen
        half_width = self.image_frame.winfo_width() // 2
        half_height = self.image_frame.winfo_height() // 2
        self.img_x = (self.img_x - half_width) * delta + half_width
        self.img_y = (self.img_y - half_height) * delta + half_height
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

    def on_closing(self):
        if self.image is not None:
            self.image.handle.close()
        self.destroy()


if __name__ == "__main__":
    app = RidgeDetectorGUI()
    app.mainloop()
