from .device import device
from .dataset import SceneTextDataset
from .model import InpaintGenerator
from . import config
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os

def performance_check(
    inpaint_gen: InpaintGenerator, 
    check_dir,
    check_img_name
):
    inpaint_gen.eval()
    with torch.no_grad(): 
        dataset = SceneTextDataset(
            cropped_txt_mask_dir=config.test_cropped_txt_mask_dir,
            cropped_bg_dir=config.test_cropped_bg_dir,
            cropped_txt_dir=config.cropped_txt_dir
        )
        data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)
        txt_img, _, _ = next(iter(data_loader))
        txt_img_backup = txt_img.detach()
        
        txt_img = txt_img.to(device)
        erased, stroke_pred_mask = inpaint_gen(txt_img)
          
        erased = denormalize_batched_tensor_to_PIL(erased)
        txt_img_backup = denormalize_batched_tensor_to_PIL(txt_img_backup)
        stroke_pred_mask = denormalize_batched_tensor_to_PIL(stroke_pred_mask, is_mask=True)
        save_img_path = os.path.join(os.path.normpath(check_dir), check_img_name)
        
        last_erased_path = os.path.join(os.path.normpath(check_dir), "last_erased.png")
        last_stroke_pred_path = os.path.join(os.path.normpath(check_dir), "last_stroke_pred.png")
        last_txt_img_path = os.path.join(os.path.normpath(check_dir), "last_txt.png")
        
        erased.save(save_img_path.replace(".png", "_text_eraise.png"))
        erased.save(last_erased_path)
        txt_img_backup.save(save_img_path.replace(".png", "_original.png"))
        txt_img_backup.save(last_txt_img_path)
        stroke_pred_mask.save(save_img_path.replace(".png", "_stroke_predict.png"))
        stroke_pred_mask.save(last_stroke_pred_path)
        
    inpaint_gen.train()

def denormalize_batched_tensor_to_PIL(tensor, is_mask=False):
    """
    input tensor is assumed to be isomorphic to [-1, 1]^N for some N.
    """
    if is_mask:
        tensor = tensor * 255
    else:
        tensor = (tensor + 1) * 127.5
    tensor = tensor[0].permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
    return Image.fromarray(tensor)


if __name__ == "__main__":
    inpaint_gen = InpaintGenerator()
    performance_check(inpaint_gen)
    