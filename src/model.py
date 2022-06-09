import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .layer import conv_bn_relu, res_blk, PartialConv2d, context_block2d, PConvAct
from .layer import CustomSequential


class SMPMEncoder(nn.Module):
    def __init__(self):
        super(SMPMEncoder, self).__init__()

        self.down1 = nn.Sequential(
            conv_bn_relu(3, 32),
            conv_bn_relu(32, 32),
        )
        self.downsize2 = nn.Conv2d(32, 64, 3, 2, padding=1)
        self.down3 = nn.Sequential(
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
        )
        self.downsize4 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.down5 = nn.Sequential(
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 128),
        )
        self.downsize6 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.down7 = nn.Sequential(
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256)
        )

        self.res_net = nn.Sequential(
            res_blk(256),
            res_blk(256),
            res_blk(256),
            res_blk(256)
        )

    def forward(self, x):
        """
        x: [n, 3, 128, 640]
        output the feature before downsizing for skip connection
        """
        x = self.down1(x)
        f0 = x

        x = self.downsize2(x)
        x = self.down3(x)
        f1 = x

        x = self.downsize4(x)
        x = self.down5(x)
        f2 = x

        x = self.downsize6(x)
        x = self.down7(x)
        x = self.res_net(x)

        return x, f0, f1, f2


class SMPMDecoder(nn.Module):
    def __init__(self):
        super(SMPMDecoder, self).__init__()
        self.up1 = nn.Sequential(
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.LeakyReLU(0.2)
        )

        self.up2 = nn.Sequential(
            conv_bn_relu(128 + 128, 128),
            conv_bn_relu(128, 128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        )

        self.up3 = nn.Sequential(
            conv_bn_relu(64 + 64, 64),
            conv_bn_relu(64, 64),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        )

        self.up4 = nn.Sequential(
            conv_bn_relu(32 + 32, 32),
            conv_bn_relu(32, 3),
            nn.Sigmoid()
        )

    def forward(self, x, f0, f1, f2):
        """
        x: [n, 256, 16, 80]
        f0:[n, 32, 128, 640]
        f1:[n, 64, 64, 320]
        f2:[n, 128, 32, 160]
        """
        x = self.up1(x)
        x = torch.cat([x, f2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.up3(x)
        x = torch.cat([x, f0], dim=1)
        x = self.up4(x)
        return x


class StrokeMaskPredictionModule(nn.Module):
    def __init__(self):
        super(StrokeMaskPredictionModule, self).__init__()
        self.encoder = SMPMEncoder()
        self.decoder = SMPMDecoder()

    def forward(self, x):
        """
        x: [n, 3, 128, 640]
        """
        feature, f0, f1, f2 = self.encoder(x)
        mask_prediction = self.decoder(feature, f0, f1, f2)
        return feature, mask_prediction


class BackgroundInpaintingModule(nn.Module):
    def __init__(self):
        super(BackgroundInpaintingModule, self).__init__()
        self.enc_1 = CustomSequential(
            PartialConv2d(3, 64, 7, 2, padding=3),
            PConvAct(64, 64, 3, 1, padding=1),
            PConvAct(64, 64, 3, 1, padding=1)
        )

        self.enc_2 = CustomSequential(
            PConvAct(64, 128, 5, 2, padding=2),
            PConvAct(128, 128, 3, 1, padding=1),
            PConvAct(128, 128, 3, 1, padding=1),
        )

        self.enc_3 = CustomSequential(
            PConvAct(128, 256, 3, 2, padding=1),
            PConvAct(256, 256, 3, 1, padding=1),
            PConvAct(256, 256, 3, 1, padding=1),
        )

        self.attention_block = context_block2d(512)
        self.aliasing = nn.Conv2d(512, 256, 3, 1, padding=1)

        self.dec_3_upcat = PConvAct(256 + 128, 256, 3, 1, upcat_first=True, activation="leaky")
        self.dec_3_conv = CustomSequential(
            PConvAct(256, 256, 3, 1, activation="leaky"),
            PConvAct(256, 256, 3, 1, activation="leaky"),
            PConvAct(256, 128, 3, 1, activation="leaky")
        )

        self.dec_2_upcat = PConvAct(128 + 64, 128, 3, 1, upcat_first=True, activation="leaky")
        self.dec_2_conv = CustomSequential(
            PConvAct(128, 128, 3, 1, activation="leaky"),
            PConvAct(128, 128, 3, 1, activation="leaky"),
            PConvAct(128, 64, 3, 1, activation="leaky")
        )

        self.dec_1_upcat = PConvAct(64 + 3, 64, 3, 1, upcat_first=True, activation="leaky")
        self.dec_1_conv = CustomSequential(
            PConvAct(64, 64, 3, 1, activation="leaky"),
            PConvAct(64, 64, 3, 1, activation="leaky"),
            PConvAct(64, 3, 3, 1, activation="tanh")
        )

    def forward(self, img, mask_pred, SMPM_feature):
        """
        x: [n, 3, 128, 640]
        """
        enc_img_0, enc_mask_0 = img, mask_pred
        img, updated_mask = self.enc_1(img, mask_pred)
        enc_img_1, enc_mask_1 = img, updated_mask  # c=64

        img, updated_mask = self.enc_2(img, updated_mask)  # downsize
        enc_img_2, enc_mask_2 = img, updated_mask  # c=128

        img, updated_mask = self.enc_3(img, updated_mask)  # downsize
        enc_img_3, enc_mask_3 = img, updated_mask  # c=256

        u = torch.cat([enc_img_3, SMPM_feature], dim=1)  # c=512
        u = self.attention_block(u)
        u = self.aliasing(u)  # c=256

        # (img, mask, enc_img=None, enc_mask=None) => img, mask
        u3, updated_mask_3 = self.dec_3_upcat(u, enc_mask_3, enc_img_2, enc_mask_2)  # c=256
        u3, updated_mask_3 = self.dec_3_conv(u3, updated_mask_3)  # c=256

        u2, updated_mask_2 = self.dec_2_upcat(u3, updated_mask_3, enc_img_1, enc_mask_1)
        u2, updated_mask_2 = self.dec_2_conv(u2, updated_mask_2)

        u1, updated_mask_1 = self.dec_1_upcat(u2, updated_mask_2, enc_img_0, enc_mask_0)
        u1, updated_mask_1 = self.dec_1_conv(u1, updated_mask_1)

        return u1, updated_mask_1

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters
        In initial training, BN set to True
        In fine-tuning stage, BN set to False
        """
        super().train(mode)
        # if not self.freeze_enc_bn:
        #     return
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                module.eval()


class InpaintGenerator(nn.Module):
    def __init__(self):
        super(InpaintGenerator, self).__init__()
        self.smpm = StrokeMaskPredictionModule()
        self.bim = BackgroundInpaintingModule()

    def forward(self, img):
        feature, mask_prediction = self.smpm(img)
        img, _ = self.bim(img, mask_prediction, feature)
        return img, mask_prediction


if __name__ == "__main__":
    smpm = StrokeMaskPredictionModule()
    bipm = BackgroundInpaintingModule()
    img = torch.randn(1, 3, 128, 640)
    feature, mask_prediction = smpm(img)

    img, maks = bipm(img, mask_prediction, feature)

    print(img.shape)
