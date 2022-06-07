from PIL import Image
from . import config


def resize_img(img):
    """
        img:  Pilow image
        """
    h, w = img.height, img.width
    ratio = config.input_height / h
    new_h, new_w = int(h * ratio), int(w * ratio)
    img = img.resize((new_w, new_h))
    return img, (h, w)


def pad_img(img):
    mode = img.mode
    new_img = Image.new(mode=mode, size=(config.input_height, config.input_width))
    new_img.paste(img)
    return new_img


def resize_and_pad_img(img, reture_hw=False):
    img, (ori_h, ori_w) = resize_img(img)
    h = img.height
    w = img.width
    padding_window = (h, w)
    img = pad_img(img)
    if not reture_hw:
        return img
    else:
        return img, padding_window, (ori_h, ori_w)
