import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import kornia

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target ,pred_normal, target_normal):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        l2_loss = (diff ** 2).mean()

        criterion = nn.CosineSimilarity(dim=0)
        cos_value = criterion(target_normal, pred_normal)
        cos_value_flatten = torch.flatten(cos_value)
        mean = torch.mean(cos_value_flatten)
        normal_loss = 1 - mean


        return 0.9* l2_loss + 0.1* normal_loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
    def forward(self, valid_mask, pred, target,pred_normal, target_normal, l_adv_r, pred_prop1, pred_prop2):
        
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        inv_valid = (valid_mask * 1) == 0
        diff_completion = target - pred
        diff_completion = diff_completion[inv_valid]
        diff_completion = diff_completion.abs().mean()


        #valid_mask = (target > -3.9934639930725098).detach() #1.8035560846328735
        diff = target - pred
        diff = diff[valid_mask]
        diff_prop_1 = target - pred_prop1
        diff_prop_2 = target - pred_prop2
        #diff_prop_1 = diff_prop_1[valid_mask]
        #diff_prop_2 = diff_prop_2[valid_mask]
        l1_loss = diff.abs().mean()
        l2_loss = (diff ** 2).mean()
        l2_loss_prop1 = (diff_prop_1 ** 2).mean()
        l2_loss_prop2 = (diff_prop_2 ** 2).mean()

        # normal_loss = torch.mean(normal_loss)
        criterion = nn.CosineSimilarity(dim=1)
        cos_value = criterion(target_normal, pred_normal)#[valid_mask[:,0,:,:]]
        cos_value_flatten = torch.flatten(cos_value)
        mean = torch.mean(cos_value_flatten)
        normal_loss = 1 - mean
        diff_normal = target_normal -  pred_normal
        #normal_loss = diff_normal.abs().mean()
        # end normal loss


        # canny = kornia.filters.Canny()
        # pred_magnitude, pred_canny = canny(pred)  # im is the pred torch tensor
        # target_magnitude, target_canny = canny(target)  # im is the pred torch tensor
        # max_pred = torch.max(pred_magnitude).item()
        # max_target = torch.max(target_magnitude).item()
        #
        # # calculating dice loss
        # top = torch.sum(torch.square(pred_magnitude)) + torch.sum(torch.square(target_magnitude))
        # bottom = 2 * torch.sum(pred_magnitude * target_magnitude)
        # dice_loss =  bottom / top
        # if torch.isnan(dice_loss) or torch.isinf(dice_loss):  # or dice_loss>40
        #     dice_loss = 0

        self.loss =  l1_loss + 0.7 * normal_loss + l_adv_r  + 0.2 * l2_loss_prop1 + 0.2 * l2_loss_prop2  #+diff_completion
        return self.loss
        
        
        
    def forward2(self, pred, target,pred_normal, target_normal):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        l1_loss = diff.abs().mean()

        # normal_loss = torch.mean(normal_loss)
        criterion = nn.CosineSimilarity(dim=1)
        cos_value = criterion(target_normal, pred_normal)
        cos_value_flatten = torch.flatten(cos_value)
        mean = torch.mean(cos_value_flatten)
        normal_loss = 1 - mean
        # end normal loss

        #start gradient map loss
        diff1 = pred - torch.roll(pred, 1, 1)
        diff2 = target - torch.roll(target, 1, 1)

        grad_map_loss = (diff2-diff1).abs().mean()


        #end gradient map loss

        self.loss = 0.85 * l1_loss + 0.1 * normal_loss + 0.05 * grad_map_loss
        

    def forward_worked_best(self, pred, target,pred_normal, target_normal):
        
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        l1_loss = diff.abs().mean()

        # normal_loss = torch.mean(normal_loss)
        criterion = nn.CosineSimilarity(dim=1)
        cos_value = criterion(target_normal, pred_normal)
        cos_value_flatten = torch.flatten(cos_value)
        mean = torch.mean(cos_value_flatten)
        normal_loss = 1 - mean
        # end normal loss

        pred_np = pred
        pred_np = pred_np.cpu()
        pred_np = pred_np.detach().numpy()
        pred_np = 256 * pred_np
        pred_np = np.uint8(pred_np)
        pred_np = cv2.Canny(pred_np, 100, 170)  # pred edge map
        # move to cuda?

        target_np = target
        target_np = target_np.cpu()
        target_np = target_np.detach().numpy()
        target_np = 256 * target_np
        target_np = np.uint8(target_np)
        target_np = cv2.Canny(target_np, 100, 170)  # pred edge map
        # move to cuda?

        top = np.sum(np.square(pred_np)) + np.sum(np.square(target_np))
        bottom = 2 * np.sum(pred_np * target_np)

        print("$$$$$$$$$$$$$$$$$$")
        print("top, bottom shape:")
        print(top.shape)
        print(bottom.shape)
        print("$$$$$$$$$$$$$$$$$$")
        dice_loss = top / bottom
        if np.isnan(dice_loss) or np.isinf(dice_loss):# or dice_loss>40:
            dice_loss = 0
        #if torch.isnan(dice_loss).any().item() and not torch.isfinite(dice_loss).all():
        #    dice_loss = 0

        self.loss = 0.7 * l1_loss + 0.3 * normal_loss #+ 0.1 * dice_loss
        return self.loss


    def forward_david(self, pred, target,pred_normal, target_normal):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        l1_loss = diff.abs().mean()

        # normal_loss = torch.mean(normal_loss)
        criterion = nn.CosineSimilarity(dim=1)
        cos_value = criterion(target_normal, pred_normal)
        cos_value_flatten = torch.flatten(cos_value)
        mean = torch.mean(cos_value_flatten)
        normal_loss = 1 - mean
        # end normal loss

        canny = kornia.filters.Canny()
        pred_magnitude, pred_canny = canny(pred)  # im is the pred torch tensor
        target_magnitude, target_canny = canny(target)  # im is the pred torch tensor

        max_pred = torch.max(pred_magnitude).item()
        max_target = torch.max(target_magnitude).item()

        pred_magnitude = (1 / (max_pred + 0.000001)) * pred_magnitude
        target_magnitude = (1 / (max_target + 0.000001)) * target_magnitude

        # calculating dice loss
        top = torch.sum(torch.square(pred_magnitude)) + torch.sum(torch.square(target_magnitude))
        bottom = 2 * torch.sum(pred_magnitude * target_magnitude)
        dice_loss = top / bottom
        if torch.isnan(dice_loss) or torch.isinf(dice_loss):  # or dice_loss>40
            dice_loss = 0


        #end gradient map loss

        self.loss = 0.8 * l1_loss + 0.1 * normal_loss + 0.1 * dice_loss
