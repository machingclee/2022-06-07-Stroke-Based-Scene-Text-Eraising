from .device import device
from .dataset import SceneTextDataset
from .model import InpaintGenerator
from . import config
from torch.utils.data import DataLoader
from PIL import Image
import torch

def performance_check(inpaint_gen: InpaintGenerator, save_img_path="results/test.png"):
    inpaint_gen.eval()
    with torch.no_grad(): 
        dataset = SceneTextDataset(
            cropped_txt_mask_dir=config.test_cropped_txt_mask_dir,
            cropped_bg_dir=config.test_cropped_bg_dir,
            cropped_txt_dir=config.cropped_txt_dir
        )
        data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=config.batch_size)
        txt_img, _, _ = next(iter(data_loader))
        txt_img_backup = txt_img.detach()
        
        txt_img = txt_img.to(device)
        img, stroke_pred_mask = inpaint_gen(txt_img)
          
        img = denormalize_batched_tensor_to_PIL(img)
        txt_img_backup = denormalize_batched_tensor_to_PIL(txt_img_backup)
        stroke_pred_mask = denormalize_batched_tensor_to_PIL(stroke_pred_mask, is_mask=True)
        
        img.save(save_img_path.replace(".png", "_text_eraise.png"))
        txt_img_backup.save(save_img_path.replace(".png", "_original.png"))
        stroke_pred_mask.save(save_img_path.replace(".png", "_stroke_predict.png"))
        
    inpaint_gen.train()

def denormalize_batched_tensor_to_PIL(tensor, is_mask=False):
    if is_mask:
        tensor = tensor * 255
    else:
        tensor = (tensor + 1) * 127.5
    tensor = tensor[0].permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
    return Image.fromarray(tensor)


if __name__ == "__main__":
    inpaint_gen = InpaintGenerator()
    performance_check(inpaint_gen)
    