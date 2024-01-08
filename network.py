import torch
import math
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import (
    create_feature_extractor,
)


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1 // divisor
    num_groups = 32 // divisor
    eps = 1e-5  # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine
    )


class Anchors(nn.Module):
    def __init__(
        self,
        stride,
        sizes=[4, 4 * math.pow(2, 1 / 3), 4 * math.pow(2, 2 / 3)],
        aspect_ratios=[0.5, 1],
        angles = [60,120,300,240]
    ):
        """
        Args:
            stride: stride of the feature map relative to the original image
            sizes: list of sizes (sqrt of area) of anchors in units of stride
            aspect_ratios: list of aspect ratios (h/w) of anchors
            angles: list of angles of anchors
        """
        super(Anchors, self).__init__()

        ##Convert to tensors
        self.sizes = torch.tensor(sizes)
        self.aspect_ratios = torch.tensor(aspect_ratios)
        self.angles = torch.tensor(angles, dtype=torch.float32)

        ##Extract lengths
        self.number_sizes = len(sizes)
        self.number_ratios = len(aspect_ratios)
        self.number_of_angles = len(angles)

        ##Calculate number of anchors
        self.number_anchors = self.number_sizes * self.number_ratios * self.number_of_angles
        self.stride = stride

        ##intilize offest tensor with shape (#anchors, #offsets = 5 [x1,y1,x2,y2,angle])
        self.anchor_offsets = torch.zeros(self.number_anchors, 5)
        
        sizes_grid, aspect_ratios_grid, angles_grid = torch.meshgrid(self.sizes, self.aspect_ratios, self.angles)

        ##Calculate the area of the anchor box
        area = sizes_grid * stride * stride

        ##Calculate the width and height based on the aspect ratio
        anchor_width = torch.sqrt(area / aspect_ratios_grid)
        anchor_height = torch.sqrt(area * aspect_ratios_grid)

        ##Calculate the indices of the anchors
        indices = torch.arange(self.number_sizes * self.number_ratios * self.number_of_angles)
        
        ##Calculate every anchor offeset
        self.anchor_offsets[indices, 0] = -anchor_width.view(-1)/2
        self.anchor_offsets[indices, 1] = -anchor_height.view(-1)/2
        self.anchor_offsets[indices, 2] = anchor_width.view(-1)/2
        self.anchor_offsets[indices, 3] = anchor_height.view(-1)/2
        self.anchor_offsets[indices, 4] = angles_grid.reshape(-1)

    def forward(self, x):
        """
        Args:
            x: feature map of shape (B, C, H, W)
        Returns:
            anchors: Anchor list: (x1, y1, x2, y2, angles), shape: (B, A*5, H, W), A = # of anchors
        
        NOTE: x1,y1,x2,y2 are rotated in the forward function
        """
        ##Extaact height, width and batch size
        height = x.size()[2]
        width = x.size()[3]
        batch = x.size()[0]

        ##Transform points using meshgrid
        range_x = torch.arange(width, device = x.device)
        range_y = torch.arange(height, device = x.device)
        mesh_x, mesh_y = torch.meshgrid(range_x,range_y,indexing="xy")

        ##Multiply by stride to account for feature map -> original size 
        mesh_y = mesh_y * self.stride
        mesh_x = mesh_x * self.stride

        ##Transfer anchor offesets the the same device as the input
        self.anchor_offsets = self.anchor_offsets.to(x.device)

        ##Add offsets the the mesh grid points to get x1,y1,x2,y2,angle
        x1 = mesh_x + self.anchor_offsets[:, 0].view(1, -1, 1, 1)/2
        y1 = mesh_y + self.anchor_offsets[:, 1].view(1, -1, 1, 1)/2
        x2 = mesh_x + self.anchor_offsets[:, 2].view(1, -1, 1, 1)/2
        y2 = mesh_y + self.anchor_offsets[:, 3].view(1, -1, 1, 1)/2
        deg_angle = self.anchor_offsets[:, 4].view(1, -1, 1, 1)
        deg_angle = deg_angle.expand(-1, -1, y2.shape[2], y2.shape[3])

        ##get cx, cy, w, h
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        anchor_width = x2 - x1
        anchor_height = y2 - y1

        ##calculate the corners
        corners_x = torch.stack([center_x - anchor_width / 2, center_x + anchor_width / 2, center_x + anchor_width / 2, center_x - anchor_width / 2])
        corners_y = torch.stack([center_y - anchor_height / 2, center_y - anchor_height / 2, center_y + anchor_height / 2, center_y + anchor_height / 2])

        ##subtract the center points from corners
        corners_x -= center_x
        corners_y -= center_y

        ##Convert degrees to radians and rotate the corners
        angle = deg_angle * math.pi / 180
        rot_corners_x = corners_x * torch.cos(angle) - corners_y * torch.sin(angle)
        rot_corners_y = corners_x * torch.sin(angle) + corners_y * torch.cos(angle)

        ##add the center points back
        rot_corners_x += center_x
        rot_corners_y += center_y

        ##Extract the new (x1, y1, x2, y2) coordinates of the anchor boxes
        x1 = rot_corners_x.min(dim=0)[0]
        y1 = rot_corners_y.min(dim=0)[0]
        x2 = rot_corners_x.max(dim=0)[0]
        y2 = rot_corners_y.max(dim=0)[0]
        
        ##stack anchors 
        anchors = torch.stack((x1,y1,x2,y2,deg_angle), dim = 2).to(x.device)
        
        ##reshape the required dimensions for return output
        anchors = anchors.view(1, self.number_anchors*5, height, width)
        
        ##repeat for every batch
        anchors = anchors.repeat(batch, 1, 1, 1)

        return anchors


class RetinaNet(nn.Module):
    def __init__(self, p67=False, fpn=False):
        super(RetinaNet, self).__init__()
        self.resnet = [
            create_feature_extractor(
                resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
                return_nodes={
                    "layer2.3.relu_2": "conv3",
                    "layer3.5.relu_2": "conv4",
                    "layer4.2.relu_2": "conv5",
                },
            )
        ]
        self.resnet[0].eval()
        ##1 classification output for Binary Classification 0/1, 24 anchors, 4 angles * 2 aspect ratios * 3 sizes = 24  anchors
        self.cls_head, self.bbox_head = self.get_heads(1, 24)

        self.p67 = p67
        self.fpn = fpn

        anchors = nn.ModuleList()

        self.p5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
            group_norm(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
        )
        self._init(self.p5)
        anchors.append(Anchors(stride=32))

        if self.p67:
            self.p6 = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p6)
            self.p7 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p7)
            anchors.append(Anchors(stride=64))
            anchors.append(Anchors(stride=128))

        if self.fpn:
            self.p4_lateral = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                group_norm(256),
            )
            self.p4 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p4)
            self._init(self.p4_lateral)
            anchors.append(Anchors(stride=16))

            self.p3_lateral = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), group_norm(256)
            )
            self.p3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p3)
            self._init(self.p3_lateral)
            anchors.append(Anchors(stride=8))

        self.anchors = anchors

    def _init(self, modules):
        for layer in modules.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def to(self, device):
        super(RetinaNet, self).to(device)
        self.anchors.to(device)
        self.resnet[0].to(device)
        return self

    def get_heads(self, num_classes, num_anchors, prior_prob=0.01):
        cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            ##output 1 class for prediction for every anchor
            nn.Conv2d(
                256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
            ),
        )
        bbox_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            ##output 5 offsets for every anchor (x1,y1,x2,y2,angle)
            nn.Conv2d(256, num_anchors * 5, kernel_size=3, stride=1, padding=1),
        )

        # Initialization
        for modules in [cls_head, bbox_head]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(cls_head[-1].bias, bias_value)

        return cls_head, bbox_head

    def get_ps(self, feats):
        conv3, conv4, conv5 = feats["conv3"], feats["conv4"], feats["conv5"]
        p5 = self.p5(conv5)
        outs = [p5]

        if self.p67:
            p6 = self.p6(conv5)
            outs.append(p6)

            p7 = self.p7(p6)
            outs.append(p7)

        if self.fpn:
            p4 = self.p4(
                self.p4_lateral(conv4)
                + nn.Upsample(size=conv4.shape[-2:], mode="nearest")(p5)
            )
            outs.append(p4)

            p3 = self.p3(
                self.p3_lateral(conv3)
                + nn.Upsample(size=conv3.shape[-2:], mode="nearest")(p4)
            )
            outs.append(p3)
        return outs

    def forward(self, x):
        with torch.no_grad():
            feats = self.resnet[0](x)

        feats = self.get_ps(feats)

        # apply the class head and box head on top of layers
        outs = []
        for f, a in zip(feats, self.anchors):
            cls = self.cls_head(f)
            bbox = self.bbox_head(f)
            outs.append((cls, bbox, a(f)))
        return outs
