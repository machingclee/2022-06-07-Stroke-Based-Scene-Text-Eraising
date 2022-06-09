from .src.device import device
from .src.model import InpaintGenerator
from .src.train import train
from .src import config
from torch.optim import Adam


def main():
    inpaint_gen = InpaintGenerator()
    inpaint_gen.to(device)
    inpaint_gen.train()
    opt = Adam(inpaint_gen.parameters(), lr=1e-4, betas=(config.beta))

    train(inpaint_gen, opt)


if __name__ == "__main__":
    main()
