from matplotlib.pyplot import bar
from src import config
from tqdm import tqdm
from torch.utils.data import DataLoader
from .dataset import SceneTextDataset
from .device import device
from .loss import get_mask_l1_loss, get_dice_loss, get_total_variation_loss, get_style_and_perceptual_loss
from torch.optim import Adam
from .model import InpaintGenerator
from .performnace import performance_check
from .utils import ConsoleLog
import torch.nn as nn
import torch
import os
import numpy as np

console_log = ConsoleLog(lines_up_on_end=1)
L1loss = nn.L1Loss()


def train(
    inpaint_gen: InpaintGenerator,
    epoches=7,
    batch_size=10,
    start_epoch=1,
    performance_check_dir="performance_check",
    start_lr=1e-4,
    last_lr=1e-5,
    beta=(0.9, 0.999),
    pths_dir="pths",
    check_performance=False
):
    dataset = SceneTextDataset()
    data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
    lr_decay = np.power(last_lr / start_lr, 1 / (epoches - 1))
    opt = Adam(inpaint_gen.parameters(), lr=start_lr, betas=(beta))

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        opt,
        gamma=lr_decay
    )

    for epoch in range(epoches):
        epoch = epoch + start_epoch
        for batch, data in enumerate(tqdm(
            data_loader, initial=1, desc=f"Epoch {epoch}",
            bar_format=config.bar_format
        )):
            batch = batch + 1

            txt_img, bg_img, txt_mask_img = data

            txt_img = txt_img.to(device)
            bg_img = bg_img.to(device)
            txt_mask_img = txt_mask_img.to(device)

            bg_predict, mask_predict = inpaint_gen(txt_img)

            mask_l1_loss = get_mask_l1_loss(mask_predict, txt_mask_img)
            dice_loss = get_dice_loss(mask_predict, txt_mask_img)

            # I_comp focus on non-hole region of bg_predict
            I_comp = mask_predict * bg_img + (1 - mask_predict) * bg_predict

            tv_loss = get_total_variation_loss(bg_predict)

            pixel_loss = (
                L1loss(mask_predict * bg_predict, mask_predict * bg_img) +
                6 * L1loss((1 - mask_predict) * bg_predict, (1 - mask_predict) * bg_img)
            )

            style_percep_loss_1 = get_style_and_perceptual_loss(bg_predict, bg_img)
            style_percep_loss_2 = get_style_and_perceptual_loss(I_comp, bg_img)
            style_percep_loss = style_percep_loss_1 + style_percep_loss_2

            loss = mask_l1_loss + dice_loss + pixel_loss + tv_loss + style_percep_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            console_log.print([
                ("loss", loss.item()),
                ("- mask_logl1_loss", mask_l1_loss.item()),
                ("- pixel_loss", pixel_loss.item()),
                ("- dice_loss", dice_loss.item()),
                ("- tv_loss", tv_loss.item()),
                ("- style_percep_loss", style_percep_loss.item()),
                ("lr", lr_scheduler.get_last_lr()[0])
            ])

            if check_performance and batch % (config.sample_result_per_n_images // batch_size) == 0:
                check_dir = performance_check_dir
                check_img_name = f"epoch_{epoch}_batch_{batch}.png"
                performance_check(inpaint_gen, check_dir, check_img_name)
        lr_scheduler.step()
        state_dict = inpaint_gen.state_dict()
        torch.save(state_dict, os.path.join(pths_dir, 'model_epoch_{}.pth'.format(epoch)))
