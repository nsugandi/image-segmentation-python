import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button, Canvas
from PIL import Image, ImageTk

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation App")

        self.image_path = None
        self.image = None
        self.segmentation_mask = None

        self.label = Label(root, text="Select an image for segmentation:")
        self.label.pack(pady=10)

        self.load_button = Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=5)

        self.segment_button = Button(root, text="Segment Image", command=self.segment_image)
        self.segment_button.pack(pady=5)

        self.canvas = Canvas(root)
        self.canvas.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.display_image()

    def segment_image(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            colored_mask = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
            self.segmentation_mask = cv2.addWeighted(self.image, 0.5, colored_mask, 0.5, 0)

            self.display_segmentation()

    def display_image(self):
        if self.image is not None:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            self.canvas.config(width=img.width(), height=img.height())
            self.canvas.create_image(0, 0, anchor='nw', image=img)
            self.canvas.image = img

    def display_segmentation(self):
        if self.segmentation_mask is not None:
            img = cv2.cvtColor(self.segmentation_mask, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            self.canvas.config(width=img.width(), height=img.height())
            self.canvas.create_image(0, 0, anchor='nw', image=img)
            self.canvas.image = img

if __name__ == "__main__":
    root = Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
