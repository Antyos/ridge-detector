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
        filemenu.add_command(label="Open Image", command=self.open_image)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

        self.img = None
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
        if self.img is None:
            return
        self.tk_img = ImageTk.PhotoImage(self.img)
        self.canvas.delete("all")
        # Always draw at (0,0) and use canvas scale/pan for zoom/pan
        self.img_canvas_id = self.canvas.create_image(
            0, 0, anchor=tk.CENTER, image=self.tk_img
        )

    def on_zoom(self, event):
        if self.img is None:
            return
        # Get mouse position in canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        # Determine zoom factor
        if hasattr(event, "delta") and event.delta:
            factor = 1 + float(np.sign(event.delta) * 0.1)
        elif hasattr(event, "num"):
            factor = 1.1 if event.num == 4 else 0.9
        else:
            factor = 1.0
        self.canvas.scale(tk.ALL, x, y, factor, factor)

    def on_pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)


if __name__ == "__main__":
    app = RidgeDetectorGUI()
    app.mainloop()
