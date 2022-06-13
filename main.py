from src.device import device
from src.model import InpaintGenerator
from src.train import train
from src import config
from torch.optim import Adam

import torch


def main():
    inpaint_gen = InpaintGenerator()
    # refactor later to get parameters from argument parser,
    # if pth_path is provided, we consider it as re-training
    pth_path = None

    if pth_path is not None:
        inpaint_gen.load_state_dict(torch.load(pth_path, map_location=device))

    inpaint_gen.to(device)
    inpaint_gen.train()

    train(
        inpaint_gen,
        epoches=20,
        start_epoch=1,
        batch_size=12,
        start_lr=1e-4,
        last_lr=1e-4,
        beta=(0.9, 0.999),
        pths_dir="pths",
        check_performance=True,
        performance_check_dir="performance_check"
    )


if __name__ == "__main__":
    main()
