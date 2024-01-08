import torch
from torch.utils.data import Dataset
import numpy as np
import json
import skimage.io
import skimage.transform
import skimage.color
import skimage
import numpy as np

from collections import defaultdict
from PIL import Image
import cv2
from pycocotools import cocoeval
from pycocotools import mask as maskUtils
from sklearn.model_selection import train_test_split

class CEPDOFDataset(Dataset):
    def __init__(self, split='train', min_sizes=[800], 
                 seed=0, transform=None):
        ##annotation and image paths
        self.annotation_file = 'CEPDOF\CEPDOF\\annotations\Lunch1.json'
        self.image = 'CEPDOF\CEPDOF\Lunch1'
       
        ##Load ground truth annotations and images
        json_dict = json.load(open(self.annotation_file))
        annotations_dict = json_dict["annotations"]
        images_dict = json_dict["images"]

        image_annotations_dict = defaultdict(list)
        for annotation in annotations_dict:
            image_annotations_dict[annotation['image_id']].append(annotation)

        ##Split the data into train, validation, and test sets (60/20/20 split)
        train_images, test_images = train_test_split(images_dict, test_size=0.2, random_state=seed)
        train_images, val_images = train_test_split(train_images, test_size=0.25, random_state=seed)

        if split == 'train':
            self.images = train_images
        elif split == 'val':
            self.images = val_images
        elif split == 'test':
            self.images = test_images
        else:
            raise ValueError("Invalid split type. Expected 'train', 'val', or 'test', but got {}".format(split))

        ##Filter the annotations based on the selected images
        self.annotation = [annotation for image in self.images for annotation in image_annotations_dict[image['id']]]

        ##intilize binary category map and classes
        cat_map = np.zeros(2, dtype=np.int64)
        classes = {0:"Background",1:"Person"}
        
        cat_map[0] = 0
        cat_map[1] = 1

        self.cat_map = cat_map
        self.classes = classes
        
        ##2 classes "background" and "person"
        self.num_classes = 2

        self.rng = np.random.RandomState(seed=seed)
        self.min_sizes = min_sizes
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def _get_annotation(self, index):
        ##Extract image id from the given index
        image_id = self.images[index]["id"]
        
        ##Extract annotations from the given image id
        filtered_annotations = [ann for ann in self.annotation if ann["image_id"] == image_id]
        
        ##remap categories to 0 or 1. In this case it will always be 1 that is the only category
        ##extract information from the annotations
        cls = np.array([a['category_id'] for a in filtered_annotations]).astype(np.int64)
        cls = self.cat_map[cls]
        is_crowd = np.array([a['iscrowd'] for a in filtered_annotations])
        anno_bboxes = np.array([a['bbox'] for a in filtered_annotations])
        image_id = image_id
        resize_factor = None

        if self.transform:
            ##Normilize the image and resize to 768x768
            image = Image.open(self.image + "/" + image_id + ".jpg")
            image = np.asarray(image, dtype=np.float32) / 255
            image, anno_bboxes, cls, is_crowd, image_id, resize_factor = self.transform([image, anno_bboxes, cls[..., np.newaxis], is_crowd, image_id])


        return image, anno_bboxes, cls, is_crowd, image_id, resize_factor

    def __getitem__(self, index):
        """
        Args:
            index: int, index of the image
        Returns:
            image: (3, H, W) tensor, normalized to [0,1], mean subtracted, std divided
            cls: (N) tensor of class indices
            bbox: (N, 5) tensor of bounding boxes in the format (x1, y1, x2, y2,degree)
            is_crowd: (N) tensor of booleans indicating whether the bounding box is a crowd
        """
        image, bboxes, cls, is_crowd, image_id, resize_factor = self._get_annotation(index)
        bboxes = bboxes[is_crowd == 0, :]
        cls = cls[is_crowd == 0]
        is_crowd = is_crowd[is_crowd == 0]
        return image, bboxes, cls, is_crowd, image_id, resize_factor

    def evaluate(self, result_file_name):
        CEPDOF_gt = json.load(open('CEPDOF\CEPDOF\\annotations\Lunch1.json', 'r'))
        CEPDOF_results = json.load(open(result_file_name, 'r'))
        catIds = sorted([cat['id'] for cat in CEPDOF_gt['categories']])
        CEPDOFevalV = CEPDOFeval(CEPDOF_gt, CEPDOF_results, 'bbox')
        CEPDOFevalV.evaluate()
        CEPDOFevalV.accumulate()
        CEPDOFevalV.summarize()
        metrics = [CEPDOFevalV.stats]
        for catId in catIds:
            CEPDOF_eval = CEPDOFeval(CEPDOF_gt, CEPDOF_results, 'bbox')
            CEPDOF_eval.params.catIds = [catId]
            CEPDOF_eval.evaluate()
            CEPDOF_eval.accumulate()
            CEPDOF_eval.summarize()
            metrics.append(CEPDOF_eval.stats)
        metrics = np.array(metrics) 
        return metrics, ['all'] + list(self.classes.values())
    
    def image_aspect_ratio(self, image_index):
        width = self.images[image_index]["width"]
        height = self.images[image_index]['height']
        return float(width) / float(height)


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    """Same as mp3 implementation except resized to 768 x 768"""
    def __call__(self, sample, min_side=768, max_side=768):
        image, bboxes, cls, is_crowd, image_id = sample[0], sample[1], sample[2], sample[3], sample[4]

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        bboxes[:, :4] *= scale

        return torch.from_numpy(new_image), torch.from_numpy(bboxes), torch.from_numpy(cls), is_crowd, image_id, scale

class Normalizer(object):
    """Same as mp3 implementation"""
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        sample[0] = ((sample[0].astype(np.float32)-self.mean)/self.std)

        return sample

class UnNormalizer(object):
    """Same as mp3 implementation"""
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """Same as mp3 implementation"""
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
 
def collater(data):
    """Altered to handle 5th degree dimension"""
    imgs = [s[0] for s in data]
    bboxes = [s[1] for s in data]
    cls = [s[2] for s in data]
    is_crowd = [s[3] for s in data]
    image_id = [s[4] for s in data]
    resize_factor = [s[5] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(c.shape[0] for c in bboxes)
    
    if max_num_annots > 0:

        bboxes_padded = torch.ones((len(bboxes), max_num_annots, 5)) * -1
        cls_padded = torch.ones((len(bboxes), max_num_annots, 1)) * -1

        if max_num_annots > 0:
            for idx in range(len(bboxes)):
                bbox = bboxes[idx]
                cl = cls[idx]
                if bbox.shape[0] > 0:
                    bboxes_padded[idx, :bbox.shape[0], :] = bbox
                    cls_padded[idx, :cl.shape[0], :] = cl
    else:
        bboxes_padded = torch.ones((len(bboxes), 1, 6)) * -1
        cls_padded = torch.ones((len(bboxes), 1, 6)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return padded_imgs, cls_padded.to(torch.int64), bboxes_padded, is_crowd, image_id, resize_factor

class CEPDOFeval(cocoeval.COCOeval):
    """Source : https://github.com/duanzhiihao/CEPDOF_tools"""
    '''
    Interface for evaluating detection on the CEPDOF dataset.
    
    The usage for CEPDOFeval is as follows (nearly identical to the COCO dataset API):
     gt_data=..., dts_data=...       # load dataset and results
     E = CocoEval(gt_data,dts_data)  # initialize CocoEval object
     E.params.recThrs = ...          # set parameters as desired
     E.evaluate()                    # run per image evaluation
     E.accumulate()                  # accumulate per image results
     E.summarize()                   # display summary metrics of results
    For example usage see https://github.com/duanzhiihao/CEPDOF_tools
    
    The evaluation parameters are as follows (defaults in brackets):
     imgIds     - [all] N img ids to use for evaluation
     catIds     - NOTE: only support category_id=1, namely, 'person'
     iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
     recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
     areaRng    - [...] A=4 object area ranges for evaluation
     maxDets    - [1 10 100] M=3 thresholds on max detections per image
     iouType    - NOTE: only support 'bbox'
    Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    
    evaluate(): evaluates detections on every image and every category and
    concats the results into the "evalImgs" with fields:
     dtIds      - [1xD] id for each of the D detections (dt)
     gtIds      - [1xG] id for each of the G ground truths (gt)
     dtMatches  - [TxD] matching gt id at each IoU or 0
     gtMatches  - [TxG] matching dt id at each IoU or 0
     dtScores   - [1xD] confidence of each dt
     gtIgnore   - [1xG] ignore flag for each gt
     dtIgnore   - [TxD] ignore flag for each dt at each IoU
    
    accumulate(): accumulates the per-image, per-category evaluation
    results in "evalImgs" into the dictionary "eval" with fields:
     params     - parameters used for evaluation
     date       - date evaluation was performed
     counts     - [T,R,K,A,M] parameter dimensions (see above)
     precision  - [TxRxKxAxM] precision for every evaluation setting
     recall     - [TxKxAxM] max recall for every evaluation setting
    Note: precision and recall==-1 for settings with no gt objects.
    
    See also eval_demo.ipynb
    '''
    def __init__(self, gt_json, dt_json, iouType='bbox'):
        '''
        Args:
            gt_json: if str, it should be the path to the annotation file.
                     if dict, it should be the annotation json.
            dt_json: if str, it should be the path to the detection file.
                     if dict, it should be the detection json.
        '''
        assert iouType == 'bbox', 'Only support (rotated) bbox iou type'
        self.gt_json = json.load(open(gt_json, 'r')) if isinstance(gt_json, str) \
                       else gt_json
        self.dt_json = json.load(open(dt_json, 'r')) if isinstance(dt_json, str) \
                       else dt_json
        self._preprocess_dt_gt()
        self.params = cocoeval.Params(iouType=iouType)
        self.params.imgIds = sorted([img['id'] for img in self.gt_json['images']])
        self.params.catIds = sorted([cat['id'] for cat in self.gt_json['categories']])
        # Initialize some variables which will be modified later
        self.evalImgs = defaultdict(list)   # per-image per-category eval results
        self.eval     = {}                  # accumulated evaluation results
    
    def _preprocess_dt_gt(self):
        # We are not using 'id' in ground truth annotations because it's useless.
        # However, COCOeval API requires 'id' in both detections and ground truth.
        # So, add id to each dt and gt in the dt_json and gt_json.
        for i, gt in enumerate(self.gt_json['annotations']):
            gt['id'] = gt.get('id', i+1)
        for i, dt in enumerate(self.dt_json):
            dt['id'] = dt.get('id', i+1)
            # Calculate the areas of detections if there is not. category_id
            dt['area'] = dt.get('area', dt['bbox'][2]*dt['bbox'][3])
            dt['category_id'] = dt.get('category_id', 1)
        # A dictionary mapping from image id to image information
        self.imgId_to_info = {img['id']:img for img in self.gt_json['images']}

    def _prepare(self):
        p = self.params
        gts = [ann for ann in self.gt_json['annotations']]
        dts = self.dt_json

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt.get('ignore', False) or gt.get('iscrowd', False)
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            raise NotImplementedError('Do not support segmentation for now')
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [convert_bbox_format(d['bbox']) for d in dt]
            # get image width and height
            img = self.imgId_to_info[imgId]
            img_size = (img['height'], img['width'])
            ious = iou_rle(d, g)
        else:
            raise Exception('unknown iouType for iou computation')
        return ious
    

def convert_bbox_format(bbox):
    """
    Convert Anchor Format [x1,y1,x2,y2,degree] to [cx,cy,w,h,degree].
    
    Args:
        bbox: list, format = [x1,y1,x2,y2,degree]

    Return:
        list, format = [cx,cy,w,h,degree]
    """
    x1, y1, x2, y2, a = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h, a]

def iou_rle(boxes1, boxes2, img_size=768):
    """Source : https://github.com/duanzhiihao/CEPDOF_tools"""
    '''
    Use mask and Run Length Encoding to calculate IOU between rotated bboxes.

    NOTE: rotated bounding boxes format is [cx, cy, w, h, degree (clockwise)].

    Args:
        boxes1: list[list[float]], shape[M,5], 5=[cx, cy, w, h, degree]
        boxes2: list[list[float]], shape[N,5], 5=[cx, cy, w, h, degree]
        img_size: int or list, (height, width)

    Return:
        ious: np.array[M,N], ious of all bounding box pairs
    '''
    assert isinstance(boxes1, list) and isinstance(boxes2, list)
    boxes1 = np.array(boxes1).reshape(-1, 5)
    boxes2 = np.array(boxes2).reshape(-1, 5)
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))
    
    # Convert angle from degree to radian
    boxes1[:,4] = boxes1[:,4] * np.pi / 180
    boxes2[:,4] = boxes2[:,4] * np.pi / 180

    # Convert [cx,cy,w,h,angle] to verticies
    b1 = xywha2vertex(boxes1, is_degree=False).tolist()
    b2 = xywha2vertex(boxes2, is_degree=False).tolist()
    
    # Calculate IoU using COCO API
    h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
    b1 = maskUtils.frPyObjects(b1, h, w)
    b2 = maskUtils.frPyObjects(b2, h, w)
    ious = maskUtils.iou(b1, b2, [0 for _ in b2])

    return ious

def iou_rle_tensor(boxes1, boxes2, img_size=768):
    """Source : https://github.com/duanzhiihao/CEPDOF_tools"""
    '''
    Use mask and Run Length Encoding to calculate IOU between rotated bboxes.

    NOTE: rotated bounding boxes format is [cx, cy, w, h, degree (clockwise)].

    Args:
        boxes1: tesor, shape[M,5], 5=[cx, cy, w, h, degree]
        boxes2: tnesor, shape[N,5], 5=[cx, cy, w, h, degree]
        img_size: int or list, (height, width)

    Return:
        ious: Tensor[M,N], ious of all bounding box pairs
    '''
    assert isinstance(boxes1, torch.Tensor) and isinstance(boxes2, torch.Tensor)
    boxes1 = boxes1.cpu().numpy() if boxes1.is_cuda else boxes1.numpy()
    boxes2 = boxes2.cpu().numpy() if boxes2.is_cuda else boxes2.numpy()
    boxes1 = boxes1.reshape(-1, 5)
    boxes2 = boxes2.reshape(-1, 5)
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))
    
    # Convert angle from degree to radian
    boxes1[:,4] = boxes1[:,4] * np.pi / 180
    boxes2[:,4] = boxes2[:,4] * np.pi / 180

    # Convert [cx,cy,w,h,angle] to verticies
    b1_np = xywha2vertex(boxes1, is_degree=False)
    b2_np = xywha2vertex(boxes2, is_degree=False)
    
    # Calculate IoU using COCO API
    h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
    b1 = maskUtils.frPyObjects(b1_np.tolist(), h, w)
    b2 = maskUtils.frPyObjects(b2_np.tolist(), h, w)
    ious = maskUtils.iou(b1, b2, [0 for _ in b2])

    return  torch.from_numpy(ious)

def xywha2vertex(box, is_degree):
    """Source : https://github.com/duanzhiihao/CEPDOF_tools"""
    '''
    Convert bounding boxes to vertices.
    NOTE: if is_degree=True, the angle will be converted to radian **in-place**.

    Args:
        box: tensor, shape(batch,5), 5 = [cx, cy, w, h, angle (clockwise)]
        is_degree: whether the angle is degree or radian

    Return:
        tensor, shape(batch,4,2): 4 = [topleft, topright, bottomright, bottomleft]
    '''
    assert box.ndim == 2 and box.shape[1] >= 5
    if is_degree:
        # convert to radian **in-place**
        box[:, 4] = box[:, 4] * np.pi / 180
    batch = box.shape[0]
    center = box[:,0:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]

    # calculate vertical vector
    verti = np.empty((batch,2))
    verti[:,0] = (h/2) * np.sin(rad)
    verti[:,1] = - (h/2) * np.cos(rad)
    # calculate horizontal vector
    hori = np.empty((batch,2))
    hori[:,0] = (w/2) * np.cos(rad)
    hori[:,1] = (w/2) * np.sin(rad)
    # calculate four vertices
    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    return np.concatenate([tl,tr,br,bl], axis=1)

def draw_cxcywhd(im, cx, cy, w, h, degree, color=(255,0,0), linewidth=5):
    """Source : https://github.com/duanzhiihao/CEPDOF_tools"""
    '''
    Draw a rotated bounding box on an np-array image in-place.

    Args:
        im: image numpy array, shape(h,w,3)
        cx, cy, w, h: the center x, center y, width, and height of the rot bbox
        degree: the angle that the bbox is rotated clockwise
    '''
    c, s = np.cos(degree/180*np.pi), np.sin(degree/180*np.pi)
    R = np.asarray([[c, s], [-s, c]])
    pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot_pts = []
    for pt in pts:
        rot_pts.append(([cx, cy] + pt @ R).astype(int))
    contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])
    cv2.polylines(im, [contours], isClosed=True, color=color,
                thickness=linewidth, lineType=cv2.LINE_AA)