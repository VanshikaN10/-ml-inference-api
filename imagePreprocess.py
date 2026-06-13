import numpy as np
from PIL import Image, ImageOps


def preprocess(image: Image.Image):                # self refers to the entire app object — the whole DigitRecognizerApp class. The image is self.image, which is one attribute stored inside that object.
    # If average pixel is bright (light background), invert
    # If average pixel is dark (dark background), leave as is
    avg_pixel = np.array(image).mean()
    if avg_pixel > 127:
        image = ImageOps.invert(image)

    bbox = image.getbbox()      # getbbox() finds the bounding box of non-black pixels — basically "where is the digit drawn?" If the canvas is empty, bbox is None, so we return early.
    if bbox is None:
        return None
    img = image.crop(bbox)      # Crops tightly around just the digit, removing empty black space around it.

    pad = 30                         # Adds black padding back around the digit. MNIST digits aren't edge-to-edge — they have breathing room. Without this, accuracy drops noticeably.
    img = ImageOps.expand(img, border=pad, fill=0)

    img.thumbnail((20, 20), Image.LANCZOS)           # Shrinks the digit to fit inside 20×20. Why 20 not 28? Because MNIST digits actually occupy a 20×20 center region inside the 28×28 image.
    new_img = Image.new("L", (28, 28), 0)
    offset_x = (28 - img.width)  // 2                # This calculates how much empty space should be left on the left and top side so the image is centered horizontally and vertically resp.
    offset_y = (28 - img.height) // 2
    new_img.paste(img, (offset_x, offset_y))         # This pastes image onto the blank 28×28 image.

    img_array = np.array(new_img) / 255.0            # Converts to numpy, divides by 255 to get values between 0–1 (same as training), and flattens 28×28 into a 784-length array because your model's input layer expects (784,).
    img_array = img_array.reshape(1, 28, 28, 1)           # reshape(1, 784) means "1 image, 784 pixels"
    return img_array                                 # These are two separate jobs deliberately. Processing the image and running the model are different responsibilities. That is why no model reference here.

