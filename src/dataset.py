from PIL import Image
from torch.utils.data import Dataset
from glob import glob
from torchvision.transforms import transforms
from PIL import Image
import random
import src.config as config
import os


class ChineseDataset(Dataset):
    def __init__(self) -> None:
        super(ChineseDataset, self).__init__()
        self.bg_imgs = glob(f"{config.bg_dir}/*.png")
        random.shuffle(self.bg_imgs)

    def __getitem__(self, index):
        bg_path = self.bg_imgs[index]
        basename = os.path.basename(bg_path)
        txt_path = f"{config.txt_dir}/{basename}"
        txt_mask_path = f"{config.txt_mask_dir}/{basename}"

        bg_img = Image.open(bg_path).convert("RGB")
        txt_img = Image.open(txt_path).convert("RGB")
        txt_mask_img = Image.open(txt_mask_path).convert("RGB")

        bg_img = resize_and_pad_img(bg_img)
        txt_img = resize_and_pad_img(txt_img)
        txt_mask_img = resize_and_pad_img(txt_mask_img)

        trans_mask = transforms.Compose([
            transforms.ToTensor()
        ])

        trans_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalize from [0, 1] to [-1, 1]
        ])

        return trans_img(txt_img), trans_img(bg_img), trans_mask(txt_mask_img)

    def __len__(self):
        return len(self.bg_imgs)


def resize_img(img):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    assert h < w, "dataset should have height smaller than width"
    ratio = config.input_height / h
    new_h, new_w = int(h * ratio), int(w * ratio)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, (w, h)


def pad_img(img):
    mode = img.mode
    new_img = Image.new(mode=mode, size=(config.input_width, config.input_height))
    new_img.paste(img)
    return new_img


def resize_and_pad_img(img, return_window=False):
    img, (ori_w, ori_h) = resize_img(img)
    w = img.width
    h = img.height
    padding_window = (w, h)
    img = pad_img(img)
    if not return_window:
        return img
    else:
        return img, padding_window, (ori_h, ori_w)


def reverse_preprocessing(img_arr, padding_window, w, h):
    img = Image.fromarray(img_arr)
    img = img.crop(padding_window)
    img = img.resize((w, h))
    return img


if __name__ == "__main__":
    dataset = ChineseDataset()
    txt_img, bg_img, txt_mask = dataset[12]

    txt_img.show()
    bg_img.show()
    txt_mask.show()
