from PIL import Image
from matplotlib import image
from torch.utils.data import Dataset
from glob import glob
from torchvision.transforms import transforms
from PIL import Image
from . import config
import albumentations as A
import random
import os
import numpy as np

torch_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalize from [0, 1] to [-1, 1]
])
torch_mask_transform = transforms.Compose([
    transforms.ToTensor()
])

albumentation_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 2), p=1),
    A.Perspective(p=0.4),
    A.Rotate(limit=10, p=0.8),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5)
    ], p=1.0),
],
    additional_targets={"image1": "image"}
)


class SceneTextDataset(Dataset):
    def __init__(
        self,
        cropped_bg_dir=config.cropped_bg_dir,
        cropped_txt_dir=config.cropped_txt_dir,
        cropped_txt_mask_dir=config.cropped_txt_mask_dir
    ) -> None:
        super(SceneTextDataset, self).__init__()
        self.cropped_bg_img_paths = glob(f"{cropped_bg_dir}/*.png")
        self.cropped_txt_dir = cropped_txt_dir
        self.cropped_txt_mask_dir = cropped_txt_mask_dir
        random.shuffle(self.cropped_bg_img_paths)

    def __getitem__(self, index):
        bg_path = self.cropped_bg_img_paths[index]
        basename = os.path.basename(bg_path)
        txt_path = f"{self.cropped_txt_dir}/{basename}"
        txt_mask_path = f"{self.cropped_txt_mask_dir}/{basename}"

        bg_img = Image.open(bg_path).convert("RGB")
        txt_img = Image.open(txt_path).convert("RGB")
        txt_mask_img = Image.open(txt_mask_path).convert("RGB")

        bg_img = resize_and_padding(bg_img)
        txt_img = resize_and_padding(txt_img)
        txt_mask_img = resize_and_padding(txt_mask_img)

        bg_img = np.array(bg_img)
        txt_img = np.array(txt_img)
        txt_mask_img = np.array(txt_mask_img)

        bg_img, txt_img, txt_mask_img = albumentation_process(bg_img, txt_img, txt_mask_img)

        return (
            torch_img_transform(txt_img),
            torch_img_transform(bg_img),
            torch_mask_transform(txt_mask_img)
        )

    def __len__(self):
        return len(self.cropped_bg_img_paths)


def albumentation_process(bg_img, txt_img, txt_mask_img):
    transformsed = albumentation_transform(image=bg_img, image1=txt_img, mask=txt_mask_img)
    bg_img = transformsed["image"]
    txt_img = transformsed["image1"]
    txt_mask_img = transformsed["mask"]
    return bg_img, txt_img, txt_mask_img


def resize_img(img):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    assert h < w, "dataset should have height smaller than width"
    ratio = config.input_height / h
    new_h, new_w = int(h * ratio), int(w * ratio)

    if new_w > config.input_width:
        ratio = config.input_width / new_w
        new_h, new_w = int(new_h * ratio), int(new_w * ratio)

    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, (w, h)


def pad_img(img):
    h = img.height
    w = img.width
    img = np.array(img)
    img = np.pad(img, pad_width=((0, config.input_height - h), (0, config.input_width - w), (0, 0)), mode="reflect")
    img = Image.fromarray(img)
    assert img.height == config.input_height
    assert img.width == config.input_width
    return img


def resize_and_padding(img, return_window=False):
    img, (ori_w, ori_h) = resize_img(img)
    w = img.width
    h = img.height
    padding_window = (w, h)
    img = pad_img(img)

    if not return_window:
        return img
    else:
        return img, padding_window, (ori_w, ori_h)


def reverse_preprocessing(img_arr, padding_window, original_wh):
    """
    img_arr is in [-1, 1]^N
    """
    (x, y) = padding_window
    (w, h) = original_wh
    img_arr = (img_arr + 1) * 127.5
    img_arr = img_arr.astype("uint8")
    img = Image.fromarray(img_arr)
    img = img.crop((0, 0, x, y))
    img = img.resize((w, h))
    return img


if __name__ == "__main__":
    dataset = SceneTextDataset()
    txt_img, bg_img, txt_mask = dataset[12]

    txt_img.show()
    bg_img.show()
    txt_mask.show()
