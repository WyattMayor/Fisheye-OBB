import torch
import torch.nn as nn
from detection_utils import compute_bbox_targets, convert_anchors_to_gt_format
from torch.nn import SmoothL1Loss

class LossFunc(nn.Module):

    def forward(self, classifications, regressions, anchors, gt_clss, gt_bboxes):

        device = classifications.device
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            targets_cls = gt_clss[j, :, :]
            targets_bbox = gt_bboxes[j, :, :]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            ##Any classification that is 1 "person" is considered a positive indicies
            positive_indices = (targets_cls > 0).view(-1)
            num_positive_anchors = positive_indices.sum()

            if num_positive_anchors == 0:
                bce = -(torch.log(1.0 - classification))
                cls_loss = bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float().to(device))
                continue

            
            ##Compute Binary Cross Entropy Loss
            targets = torch.zeros(classification.shape)
            targets = targets.to(device)
            targets[positive_indices, :] = 1
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = bce
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            ##Compute Smooth L1 Loss
            loss_fn = SmoothL1Loss()
            targets_bbox = targets_bbox[positive_indices, :]
            anchor_converted = convert_anchors_to_gt_format(anchor[positive_indices, :].reshape(-1,5))
            bbox_reg_target = compute_bbox_targets(anchor_converted, targets_bbox.reshape(-1,5))
            targets = bbox_reg_target.to(device)
            regression_diff = torch.abs(targets - regression[positive_indices, :]).to(device)
            mask = torch.ne(targets, -1.0)
            regression_diff_masked = torch.where(mask, regression_diff, torch.zeros(regression_diff.shape).to(device))
            regression_loss = loss_fn(regression_diff_masked, torch.zeros_like(regression_diff_masked))
            regression_losses.append(regression_loss.mean())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
