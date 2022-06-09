from PIL import Image
from torch.utils.data import Dataset
from glob import glob
from torchvision.transforms import transforms
from PIL import Image
from . import config
import random
import os

prefeed_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalize from [0, 1] to [-1, 1]
])
prefeed_mask_transform = transforms.Compose([
    transforms.ToTensor()
])


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

        return (
            prefeed_img_transform(txt_img),
            prefeed_img_transform(bg_img),
            prefeed_mask_transform(txt_mask_img)
        )

    def __len__(self):
        return len(self.cropped_bg_img_paths)


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
    mode = img.mode
    new_img = Image.new(mode=mode, size=(config.input_width, config.input_height))
    new_img.paste(img)
    return new_img


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


def reverse_preprocessing(img_arr, padding_window, origin_wh):
    """
    img_arr is in [-1, 1]^N
    """
    (x, y) = padding_window
    (w, h) = origin_wh
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
