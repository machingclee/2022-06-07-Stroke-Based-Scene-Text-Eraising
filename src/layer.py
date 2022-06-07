import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import vgg16
from torchsummary import summary
from src.device import device


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_bn_relu, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)]
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class res_blk(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(res_blk, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1),
            nn.Conv2d(dim // 4, dim // 4, kernel_size, 1, padding=1),
            nn.Conv2d(dim // 4, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2)
        )
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        input = x
        x = self.net(x)
        x = x + input
        x = self.bn(x)
        x = self.relu(x)
        return x


class PartialConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        # Inherit the parent class (Conv2d)
        super(PartialConv2d, self).__init__(in_channels, out_channels,
                                            kernel_size, stride=stride,
                                            padding=padding, dilation=dilation,
                                            groups=groups, bias=bias,
                                            padding_mode=padding_mode)
        # Define the kernel for updating mask
        self.mask_kernel = torch.ones(self.out_channels, self.in_channels,
                                      self.kernel_size[0], self.kernel_size[1])
        # Define sum1 for renormalization
        self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] \
                                              * self.mask_kernel.shape[3]
        # Define the updated mask
        self.update_mask = None
        # Define the mask ratio (sum(1) / sum(M))
        self.mask_ratio = None
        # Initialize the weights for image convolution
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, img, mask):
        with torch.no_grad():
            if self.mask_kernel.type() != img.type():
                self.mask_kernel = self.mask_kernel.to(img)
            # Create the updated mask
            # for calcurating mask ratio (sum(1) / sum(M))
            self.update_mask = F.conv2d(mask, self.mask_kernel,
                                        bias=None, stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=1)
            # calcurate mask ratio (sum(1) / sum(M))
            self.mask_ratio = self.sum1 / (self.update_mask + 1e-8)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # calcurate WT . (X * M)
        conved = torch.mul(img, mask)
        conved = F.conv2d(conved, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        if self.bias is not None:
            # Maltuply WT . (X * M) and sum(1) / sum(M) and Add the bias
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(conved - bias_view, self.mask_ratio) + bias_view
            # The masked part pixel is updated to 0
            output = torch.mul(output, self.mask_ratio)
        else:
            # Multiply WT . (X * M) and sum(1) / sum(M)
            output = torch.mul(conved, self.mask_ratio)

        return output, self.update_mask


class CustomSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(CustomSequential, self).__init__(*args, **kwargs)

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, val=0)
        m[-1].inited = True
    else:
        nn.init.constant_(m.weight, val=0)
        m.inited = True


class context_block2d(nn.Module):
    def __init__(self, dim, ratio=4):
        super(context_block2d, self).__init__()
        self.dim = dim

        self.conv_mask = nn.Conv2d(dim, 1, kernel_size=1)  # context Modeling
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.dim, self.dim // ratio, kernel_size=1),
            nn.LayerNorm([self.dim // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // ratio, self.dim, kernel_size=1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_mask.weight, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W] 添加一个维度
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)  # softmax操作
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        out = x
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term
        return out


class UpsampleConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, dec_feature, dec_mask, enc_feature, enc_mask):
        out = self.upsample(dec_feature)
        out = torch.cat([out, enc_feature], dim=1)
        out_mask = self.upsample(dec_mask)
        out_mask = torch.cat([out_mask, enc_mask], dim=1)
        return out, out_mask


class PConvAct(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        bn=True,
        activation='relu',
        upcat_first=False,
        conv_bias=False,
    ):
        super().__init__()
        self.partialconv = PartialConv2d(in_ch, out_ch, kernel_size, stride, padding, bias=conv_bias)
        assert activation in ["relu", "leaky", "tanh"], "activation can either be \"relu\", \"leaky\" or \"tanh\""

        # Define other layers
        if upcat_first:
            self.upcat = UpsampleConcat()
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    # define overloading
    def forward(self, img, mask, enc_img=None, enc_mask=None):
        if hasattr(self, 'upcat'):
            out, update_mask = self.upcat(img, mask, enc_img, enc_mask)
            out, update_mask = self.partialconv(out, update_mask)
        else:
            out, update_mask = self.partialconv(img, mask)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)

        return out, update_mask


class Normalization(nn.Module):
    def __init__(self, vgg_mean, vgg_std):
        super(Normalization, self).__init__()
        # .view the vgg_mean and vgg_std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.

        self.vgg_mean = torch.tensor(vgg_mean).view(-1, 1, 1)
        self.vgg_std = torch.tensor(vgg_std).view(-1, 1, 1)

    def forward(self, input):
        # normalize img
        if self.vgg_mean.type() != input.type():
            self.vgg_mean = self.vgg_mean.to(input)
            self.vgg_std = self.vgg_std.to(input)
        # output in [-1, 1] -> [0, 1] -> ([0, 1] - mean) / std
        return ((input * + 1) * 0.5 - self.vgg_mean) / self.vgg_std


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_mean = [0.485, 0.456, 0.406]
        self.vgg_std = [0.229, 0.224, 0.225]
        vgg = vgg16(pretrained=True).to(device)
        normalization = Normalization(self.vgg_mean, self.vgg_std)
        self.enc_1 = nn.Sequential(normalization, *vgg.features[:5])
        self.enc_2 = nn.Sequential(*vgg.features[5:10])
        self.enc_3 = nn.Sequential(*vgg.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        feature_maps = [input]
        for i in range(3):
            feature_map = getattr(self, 'enc_{}'.format(i + 1))(feature_maps[-1])
            feature_maps.append(feature_map)
        return feature_maps[1:]
