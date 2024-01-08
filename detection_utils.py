import torch
import numpy as np
import random
from dataset import iou_rle_tensor

def set_seed(seed):
    """From MP3"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
  
def get_detections(outs):
    """Coverted to handle anchor boxes with 5 dimensions and binary classification"""
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 5
    num_classes = 1
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 5))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 5))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def convert_anchors_to_gt_format(anchors):
    '''
    Convert [x1,y1,x2,y2,degree] bounding boxes to [cx,cy,w,h,degree] format.

    Args:
        box: tensor, shape(batch,5), 5 = [x1,y1,x2,y2,degree]
    Returns:
        Bbox: tensor, shape(batch,5), 5 = [cx,cy,w,h,degree]
    '''
    # Extract the angle and convert it from degrees to radians
    degree = anchors[:, 4]

    # Calculate the center of the rectangle
    cx = (anchors[:, 0] + anchors[:, 2]) / 2
    cy = (anchors[:, 1] + anchors[:, 3]) / 2

    # Calculate the width and height of the rectangle before rotation
    w = torch.abs(anchors[:, 2] - anchors[:, 0])
    h = torch.abs(anchors[:, 3] - anchors[:, 1])

    # Construct ground truth format: (cx, cy, w, h, degree)
    gt_format_boxes = torch.stack([cx, cy, w, h, degree], dim=-1)
    return gt_format_boxes

def convert_gt_boxes(box, is_degree):
    '''
    Convert [cx,cy,w,h,degree] bounding boxes to [x1,y1,x2,y2,degree] format.

    Args:
        box: tensor, shape(batch,5), 5 = [cx,cy,w,h,degree]
        is_degree: whether the angle is degree or radian
    Returns:
        Bbox: tensor, shape(batch,5), 5 = [x1,y1,x2,y2,degree]
    '''
    ##save degree for later
    degree = box[:, 4]
    if is_degree:
        ##convert to angle from degrees to radians
        box[:, 4] = box[:, 4] * np.pi / 180
    batch = box.shape[0]
    center = box[:,0:2]
    width = box[:,2]
    height = box[:,3]
    rad = box[:,4]

    #calculate vertical vector
    vertical = torch.empty((batch,2)).to(torch.device('cuda:0'))
    vertical[:,0] = (height/2) * torch.sin(rad)
    vertical[:,1] = - (height/2) * torch.cos(rad)
    
    #calculate horizontal vector
    horizontal = torch.empty((batch,2)).to(torch.device('cuda:0'))
    horizontal[:,0] = (width/2) * torch.cos(rad)
    horizontal[:,1] = (width/2) * torch.sin(rad)
    
    ##Calculate the four corners
    top_left = center + vertical - horizontal
    top_right = center + vertical + horizontal
    bottom_right = center - vertical + horizontal
    bottom_left = center - vertical - horizontal

    ##stack all of the corners
    corners = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=1)

    ##Calculate x1, y1, x2, y2
    x_min = torch.min(corners[..., 0], dim=1).values.view(-1, 1)
    y_min = torch.min(corners[..., 1], dim=1).values.view(-1, 1)
    x_max = torch.max(corners[..., 0], dim=1).values.view(-1, 1)
    y_max = torch.max(corners[..., 1], dim=1).values.view(-1, 1)
    
    ##Use previously saved degree
    angle = degree.view(-1, 1)

    return torch.cat([x_min, y_min, x_max, y_max, angle], dim=1)

def compute_targets(anchor, bbox):
    """
    Args:
        anchors: [x1,y1,x2,y2,degree], shape: (B, A, 5), B = batch size, A = number of anchors
        bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 5)
    Returns:
        gt_clss: groundtruth class targets of shape (B, A, 1)
        gt_bboxes: groundtruth bbox targets of shape (B, A, 5)
    """
    ##extract information from input
    batch = anchor.shape[0]
    number_anchors = anchor.shape[1]

    ##Initilize return values
    gt_clss = torch.zeros((batch, number_anchors, 1),dtype=torch.int64).to(anchor.device)
    gt_bboxes = torch.zeros((batch, number_anchors, 5), dtype=torch.float32).to(anchor.device)

    for x in range(0,batch):
        ##convert anchors to the ground truth format [cx,cy,w,h,degree]
        anchor_converted = convert_anchors_to_gt_format(anchor[x])

        ##get IOUs for the batch
        ious = iou_rle_tensor(anchor_converted, bbox[x])
        maxIOU , maxIND = torch.max(ious, dim=1)

        ##Case 1 - IOU < .5
        gt_clss[x][maxIOU < 0.5] = 0
        gt_bboxes[x][maxIOU < 0.5] = 0

        ##Case 2 - IOU >= .5
        gt_clss[x][maxIOU >= 0.5] = 1
        gt_bboxes[x][maxIOU >= 0.5] = bbox[x][maxIND[maxIOU >= 0.5]]

    
    return gt_clss, gt_bboxes

def compute_bbox_targets(anchors, gt_bboxes):
    """
    Args:
        anchors: anchors of shape (A, 5), format: [cx,cy,w,h,degree]
        gt_bboxes: groundtruth object classes of shape (A, 5), format: [cx,cy,w,h,degree]
    Returns:
        bbox_reg_target: regression offset of shape (A, 5)
    """
    ##extract information from inputs
    anchor_center_x = anchors[:,0] 
    anchor_center_y = anchors[:,1] 
    gt_bbox_center_x = gt_bboxes[:,0] 
    gt_bbox_center_y = gt_bboxes[:,1] 
    anchor_width = anchors[:,2]
    anchor_height = anchors[:,3]
    gt_bbox_width = gt_bboxes[:,2]
    gt_bbox_height = gt_bboxes[:,3]

    ##calculate deltas
    delta_x = (gt_bbox_center_x - anchor_center_x) / anchor_width
    delta_y = (gt_bbox_center_y - anchor_center_y) / anchor_height
    delta_w = torch.log(torch.maximum((gt_bbox_width) , torch.tensor(1)) / anchor_width)    
    delta_h = torch.log(torch.maximum((gt_bbox_height) , torch.tensor(1)) / anchor_height) 
    delta_angle = gt_bboxes[:, 4] - anchors[:, 4]

    return torch.stack([delta_x, delta_y, delta_w, delta_h, delta_angle], dim=-1)

def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        boxes: (N, 5) tensor of (x1, y1, x2, y2, angle(degrees))
        deltas: (N, 5) tensor of (dxc, dyc, dlogw, dlogh, dangle(degrees))
    Returns
        boxes: (N, 5) tensor of (x1, y1, x2, y2, angle(degrees))
    """
    ##extract points
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    angle = boxes[:, 4]

    ##extract deltas
    dxc = deltas[:, 0]
    dyc = deltas[:, 1]
    dlogw = deltas[:, 2]
    dlogh = deltas[:, 3]
    dangle = deltas[:, 4]

    ##calculate boxes
    temp1 = (x1 + x2)/ 2 + dxc * (x2 - x1)
    temp2 = (y1 + y2)/ 2 + dyc * (y2 - y1)
    temp3 = torch.exp(dlogw) * (x2 - x1)
    temp4 = torch.exp(dlogh) * (y2 - y1)

    bbx1 = temp1 - .5 * temp3
    bby1 = temp2 - .5 * temp4
    bbx2 = temp1 + .5 * temp3
    bby2 = temp2 + .5 * temp4
    new_angle = angle + dangle

    new_boxes = torch.stack((bbx1, bby1, bbx2, bby2, new_angle), dim=1)
    return new_boxes

def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 5) tensor of (x1, y1, x2, y2, angle(degrees))
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    """
    ##convert bboxes(anchors) to gt format
    bboxes = convert_anchors_to_gt_format(bboxes)

    ##Sort indicies of scores
    IND = torch.argsort(scores, descending=True)

    
    ##Reorganize BBOX and Scores based on indicies
    bboxes = bboxes[IND]

    scores = scores[IND]

    ##Intilize Return Value
    keep = []

    ##Remove boxes fromm bboxes until you are left with 0
    while bboxes.shape[0] > 1:
        ## get the highest score bbox
        HighestScoreBox = bboxes[0]

        ##add the indicie to the return list
        keep.append(IND[0])

        ##compute the iou with respect the the highest score box
        ious = iou_rle_tensor(HighestScoreBox.unsqueeze(0), bboxes[1:])
        ious = ious.squeeze(0)

        ##No Overlapping Indicies
        NOIND = torch.nonzero(ious < threshold).squeeze(1)

        bboxes = bboxes[NOIND+1]
        scores = scores[NOIND+1]
        IND = IND[NOIND+1]

    ##convert to tensor for valid return value
    keep = torch.tensor(keep, dtype = torch.long)
    return keep
