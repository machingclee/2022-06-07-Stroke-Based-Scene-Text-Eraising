from torchvision.models import vgg16
from torchvision.transforms import Normalize
from torchsummary import summary
import torch
import torch.nn as nn

def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


class Normalization(nn.Module):
    def __init__(self, mean, std, vgg_mean, vgg_std):
        super(Normalization, self).__init__()
        # .view the vgg_mean and vgg_std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.vgg_mean = torch.tensor(vgg_mean).view(-1, 1, 1)
        self.vgg_std = torch.tensor(vgg_std).view(-1, 1, 1)

    def forward(self, input):
        # normalize img
        if self.vgg_mean.type() != input.type():
            self.mean = self.mean.to(input)
            self.std = self.std.to(input)
            self.vgg_mean = self.vgg_mean.to(input)
            self.vgg_std = self.vgg_std.to(input)
        return ((input * self.std + self.mean) - self.vgg_mean) / self.vgg_std


class VGG16FeatureExtractor(nn.Module):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    vgg_mean = [0.485, 0.456, 0.406]
    vgg_std = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg16 = vgg16(pretrained=True)
        normalization = Normalization(self.mean, self.std, self.vgg_mean, self.vgg_std)
        self.enc_1 = nn.Sequential(normalization, *vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{}'.format(i+1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        feature_maps = [input]
        for i in range(3):
            feature_map = getattr(self, 'enc_{}'.format(i+1))(feature_maps[-1])
            feature_maps.append(feature_map)
        return feature_maps[1:]


def get_