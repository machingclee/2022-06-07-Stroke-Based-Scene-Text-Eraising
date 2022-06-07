import torch
import torch.nn as nn
from src.layer import VGG16FeatureExtractor

L1loss = nn.L1Loss()
feature_extractor = VGG16FeatureExtractor()


def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def get_mask_l1_loss(mask_predict, mask_gt):
    return L1loss(mask_predict, mask_gt)


def get_dice_loss(mask_predict, mask_gt):
    epsilon = 1e-8
    intersection = (mask_predict * mask_gt).sum()  # N, H*W -> N, 1 -> scolar
    union = (mask_predict + mask_gt).sum()
    dice_loss = 1. - 2. * intersection / (union + epsilon)
    return dice_loss


def get_total_variation_loss(img):
    loss_h = L1loss(img[:, :, 1:, :], img[:, :, :-1, :])
    loss_w = L1loss(img[:, :, :, 1:], img[:, :, :, :-1])
    return loss_h + loss_w


def get_style_and_perceptual_loss(img_1, img_2):
    feat_1 = feature_extractor(img_1)
    feat_2 = feature_extractor(img_2)
    for i in range(3):
        perceptual_loss = L1loss(feat_1[i], feat_2[i])
        style_loss = L1loss(gram_matrix(feat_1[i]), gram_matrix(feat_2[i]))
    return perceptual_loss + style_loss
