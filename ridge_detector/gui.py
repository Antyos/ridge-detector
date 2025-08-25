import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk


class RidgeDetectorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ridge Detector GUI")
        self.geometry("900x600")

        # Main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Image display
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.image_frame, bg="gray", width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # For canvas-based zoom and pan
        self.canvas.bind("<MouseWheel>", self.on_zoom)  # Windows
        self.canvas.bind("<Button-4>", self.on_zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)  # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_move)

        # Open image
        self.bind_all("<Control-o>", lambda event: self.open_image())

        # Right: Parameters section (empty for now)
        self.param_frame = ttk.Frame(self.main_frame, width=300)
        self.param_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.param_frame.pack_propagate(False)

        param_label = ttk.Label(
            self.param_frame,
            text="Ridge Detector Parameters",
            font=("Arial", 14, "bold"),
        )
        param_label.pack(pady=10)

        # Menu
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Image (Ctrl+O)", command=self.open_image)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

        self.img = None
        self.img_x = 0
        self.img_y = 0
        self.img_scale = 1.0
        self.tk_img = None
        self.img_canvas_id = None

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
        self.img = img
        self.display_image()

    def display_image(self):
        # NOTE: Using PIL for scalling is not very performant for large images,
        # because pixels outside the viewbox aren't culled.
        if self.img is None:
            return
        scaled_image = self.img.resize(
            (
                int(self.img.width * self.img_scale),
                int(self.img.height * self.img_scale),
            ),
            Image.Resampling.NEAREST,
        )
        self.tk_img = ImageTk.PhotoImage(scaled_image)
        self.canvas.delete("all")
        # Always draw at (0,0) and use canvas scale/pan for zoom/pan
        self.img_canvas_id = self.canvas.create_image(
            self.img_x, self.img_y, anchor=tk.NW, image=self.tk_img
        )

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
