import numpy as np
from PIL import Image
from run import data_params, model_params

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch import nn, Tensor
from torch.autograd import Variable

import pytorch_lightning as pl

from munkres import Munkres

from abc import abstractmethod
from torchvision.models.resnet import conv3x3
from typing import List, Callable, Any, Optional

from torch.nn.modules.utils import _pair
from torch.jit.annotations import BroadcastingList2
from torchvision.extension import _assert_has_ops

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# VAE Interface
class BaseVAE(pl.LightningModule):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


# ResNet block
class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1,
                bias=False,
            )
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# RoI layer
def roi_align(
    input: Tensor,
    boxes: Tensor,
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN
    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """
    _assert_has_ops()
    # check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    return torch.ops.torchvision.roi_align(
        input,
        rois,
        spatial_scale,
        output_size[0],
        output_size[1],
        sampling_ratio,
        aligned,
    )


def _cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    # TODO add back the assert
    # assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = _cat([b for b in boxes], dim=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(torch.full_like(b[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def check_roi_boxes_shape(boxes: Tensor):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            assert (
                _tensor.size(1) == 4
            ), "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]"
    elif isinstance(boxes, torch.Tensor):
        assert (
            boxes.size(1) == 5
        ), "The boxes tensor shape is not correct as Tensor[K, 5]"
    else:
        assert False, "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"
    return


class RoIAlign(nn.Module):
    """
    See roi_align
    """

    def __init__(
        self,
        output_size: BroadcastingList2[int],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return roi_align(
            input,
            rois,
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr


# Bounding box processing
# Intersection over union
def bb_IoU(a: Tensor, b: Tensor, epsilon=1e-5) -> Tensor:
    """
    :param boxA, boxB : (Tensor) [x1, y1, x2, y2]
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = x2 - x1
    height = y2 - y1
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)

    # return the intersection over union value
    return iou


def compute_overlaps(boxes1, boxes2, epsilon=1e-5):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, 0] + b2_area[:, 0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / (union + epsilon)
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps


# Union
def bb_union(boxA: Tensor, boxB: Tensor) -> Tensor:
    """
    :param boxA, boxB : (Tensor) [x1, y1, x2, y2]
    """
    # determine the (x, y)-coordinates of the union rectangle
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    # return the union rectangle bb
    union_bb = torch.tensor([xA, yA, xB, yB])
    return union_bb


# Bounding box matching
def bb_matching(bbox_pro: Tensor, bbox_mask_rcnn: Tensor, last_id: Tensor):
    """
    param: bbox_pro: predicted bb from proposal branch [N, 4]
    param: bbox_mask_rcnn: predicted bb from mask rcnn [N, 4]
    param: last_id: instance id of bbox_pro aka of last frame
    return: Tensor: final bb list
    return: List: of bb ids
    """
    assert len(last_id) == len(
        bbox_pro
    ), "len(last_id) should be equal to len(bbox_pro), missing bbox_pro"

    last_id = [i.item() for i in last_id]
    max_id = max(last_id)
    min_id = min(last_id)
    id_num = len(last_id)

    m = Munkres()
    size = max(len(bbox_pro), len(bbox_mask_rcnn))
    matrix = np.zeros(shape=(size, size))

    for i in range(len(bbox_pro)):
        for j in range(len(bbox_mask_rcnn)):
            matching_score = bb_IoU(
                bbox_pro[i], bbox_mask_rcnn[j]
            )  # 0: mismatch -> 1: perfect match
            matching_cost = 1 - matching_score
            matrix[i][j] = matching_cost

    matrix = matrix.tolist()

    # calculating the matching cost of each bb pairs
    union_bb_list = []
    union_id = []
    bb_mask_rcnn_matched = []

    indexes = m.compute(matrix)
    for i, j in indexes:
        if i < len(bbox_pro) and j < len(bbox_mask_rcnn):
            cost = matrix[i][j]
            if cost < model_params["MATCHING_COST_THRESHOLD"] and cost > 0:
                boxes_union = bb_union(bbox_pro[i], bbox_mask_rcnn[j])
                union_bb_list.append(boxes_union.to(device))

                union_id.append(last_id[i])
                bb_mask_rcnn_matched.append(j)

    for bbox_idx in range(len(bbox_mask_rcnn)):
        if bbox_idx in bb_mask_rcnn_matched:
            continue
            # else add to bb final list
        else:
            union_bb_list.append(bbox_mask_rcnn[bbox_idx])
            max_id = max_id + 1
            # if len(bbox_pro) == len(bbox_mask_rcnn)
            union_id.append(max_id)

    if len(union_bb_list) == 0:
        union_bb_list = [
            Variable(
                torch.zeros(
                    4,
                ),
                requires_grad=False,
            ).to(device)
        ]

    return torch.stack(union_bb_list).to(device), torch.tensor(union_id).to(device)


def batch_bb_matching(
    bbox_pro: List[Tensor], bbox_mask_rcnn: List[Tensor], last_id: List[Tensor]
):
    batch_bbs = []
    batch_id = []
    for i, j, k in zip(bbox_pro, bbox_mask_rcnn, last_id):
        bbs, id = bb_matching(i, j, k)
        batch_bbs.append(bbs)
        batch_id.append(id)


# Other
def tensor_2_list_of_tensors(x: Tensor):
    x = x.tolist()
    x = [torch.tensor(i) for i in x]
    return x


# Bounding box regression
def box_refinement(box, gt_box):
    """Compute refinement delta needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y = center_y + deltas[:, 0] * height
    center_x = center_x + deltas[:, 1] * width
    height = height * torch.exp(deltas[:, 2])
    width = width * torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


# Regression bounding box post process
def pro_postprocess(pred, target):
    """The model could predict the wrong correct number of instances in image,
    which lead to unmatch dim of input and target when calculating loss.
    Here IoU overlap will be calculated and prediction and gt will be rearranged
    After squeeze:
    pro_pred_bbox: [l, 4]

    target_bbox: [m, 4]
    """
    pro_pred_bbox, pro_rois = pred
    target_bbox = target.squeeze(0)

    pro_overlaps = compute_overlaps(pro_rois, target_bbox)
    # choose matched index
    pro_roi_iou_max = torch.max(pro_overlaps, 1)[0]
    pro_positive_roi_bool = pro_roi_iou_max >= 0.5
    # deal with positive Roi
    if torch.nonzero(pro_positive_roi_bool).size():
        positive_indices = torch.nonzero(pro_positive_roi_bool)[:, 0]
        # choose positive prediction bbox
        rand_idx = torch.randperm(positive_indices.size()[0])
        positive_indices = positive_indices[rand_idx]
        pro_positive_count = positive_indices.size()[0]
        pro_positive_rois = pro_rois[positive_indices, :]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = pro_overlaps[positive_indices, :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        pro_target_bbox = target_bbox[roi_gt_box_assignment, :]

        if len(pro_target_bbox) != 0:
            pro_deltas_target = (
                box_refinement(pro_positive_rois, pro_target_bbox).detach().to(device)
            )
        else:
            pro_deltas_target = torch.FloatTensor(0, 4).to(device)
        pro_deltas_positive_pred = pro_pred_bbox[positive_indices, :]
    else:
        pro_positive_count = 0

    # deal with negative Roi
    pro_negative_roi_bool = pro_roi_iou_max < 0.5
    if torch.nonzero(pro_negative_roi_bool).size():
        negative_indices = torch.nonzero(pro_negative_roi_bool)[:, 0]
        rand_idx = torch.randperm(negative_indices.size()[0])
        negative_indices = negative_indices[rand_idx]

        pro_negative_count = negative_indices.size()[0]

        pro_negative_rois = pro_rois[negative_indices.data, :]
        pro_deltas_negative_pred = pro_pred_bbox[negative_indices.data, :]
    else:
        pro_negative_count = 0

    # Picking positive Roi
    if pro_positive_count > 0 and pro_negative_count > 0:
        pro_pred_bbox = torch.cat((pro_positive_rois, pro_negative_rois), dim=0)
        pro_deltas_pred_bbox = torch.cat(
            (pro_deltas_positive_pred, pro_deltas_negative_pred), dim=0
        )
        zeros = (
            Variable(torch.zeros(pro_negative_count, 4), requires_grad=False)
            .int()
            .to(device)
        )
        pro_deltas_target_bbox = torch.cat([pro_deltas_target, zeros], dim=0)
    elif pro_positive_count > 0:
        pro_deltas_pred_bbox = pro_deltas_positive_pred
        pro_deltas_target_bbox = pro_deltas_target

    elif pro_negative_count > 0:
        pro_deltas_pred_bbox = pro_deltas_negative_pred

        zeros = (
            Variable(torch.zeros(pro_negative_count, 4), requires_grad=False)
            .int()
            .to(device)
        )
        pro_deltas_target_bbox = torch.cat([pro_deltas_target, zeros], dim=0)
    else:
        pro_deltas_pred_bbox = Variable(torch.FloatTensor(), requires_grad=False).to(
            device
        )
        pro_deltas_target_bbox = Variable(torch.FloatTensor(), requires_grad=False).to(
            device
        )

    return pro_deltas_pred_bbox, pro_deltas_target_bbox


def aug_postprocess(pred, target):
    """The model could predict the wrong correct number of instances in image,
    which lead to unmatch dim of input and target when calculating loss.
    Here IoU overlap will be calculated and prediction and gt will be rearranged
    After squeeze:
    aug_pred_bbox: [n, 4]
    pred_logits_class: [n, 5] after max should be [n, ]
    pred_masks : [n, 24, 24]
    pred_track_ids: [n, ]

    target_bbox: [m, 4]
    target_class: [m, ]
    target_masks: [m, 24, 24]
    target_track_ids = [m, ]
    """

    aug_pred_bbox, pred_logits_class, pred_masks, aug_rois = pred
    target_bbox, target_class, target_masks = target

    target_class, target_bbox, target_masks = (
        target_class.squeeze(0),
        target_bbox.squeeze(0),
        target_masks.squeeze(0),
    )

    aug_overlaps = compute_overlaps(aug_rois, target_bbox)

    # choose matched index
    aug_roi_iou_max = torch.max(aug_overlaps, 1)[0]

    aug_positive_roi_bool = aug_roi_iou_max >= 0.5

    if torch.nonzero(aug_positive_roi_bool).size():
        positive_indices = torch.nonzero(aug_positive_roi_bool)[:, 0]
        # choose positive prediction bbox
        rand_idx = torch.randperm(positive_indices.size()[0])
        positive_indices = positive_indices[rand_idx]
        gt_positive_indices = positive_indices.clone().detach()
        aug_positive_count = positive_indices.size()[0]
        aug_positive_rois = aug_rois[positive_indices, :]
        aug_deltas_positive_pred = aug_pred_bbox[positive_indices, :]
        positive_pred_logits_class = pred_logits_class[positive_indices, :]
        positive_pred_masks = pred_masks[positive_indices, :, :]

        # choose positive target
        positive_overlaps = aug_overlaps[gt_positive_indices]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        aug_target_bbox = target_bbox[roi_gt_box_assignment, :]
        target_class = target_class[roi_gt_box_assignment]
        target_masks = target_masks[roi_gt_box_assignment, :, :]

        if len(aug_target_bbox) != 0:
            aug_deltas_target = (
                box_refinement(aug_positive_rois, aug_target_bbox).detach().to(device)
            )
        else:
            aug_deltas_target = torch.FloatTensor(0, 4).to(device)

    else:
        aug_positive_count = 0

    aug_negative_roi_bool = aug_roi_iou_max < 0.5

    if torch.nonzero(aug_negative_roi_bool).size():
        negative_indices = torch.nonzero(aug_negative_roi_bool)[:, 0]
        rand_idx = torch.randperm(negative_indices.size()[0])
        negative_indices = negative_indices[rand_idx]
        aug_negative_count = negative_indices.size()[0]
        aug_negative_rois = aug_rois[negative_indices, :]
        aug_deltas_negative_pred = aug_pred_bbox[negative_indices, :]

        aug_pred_negative_logits_class = pred_logits_class[negative_indices, :]
        aug_pred_negative_masks = pred_masks[negative_indices, :, :]

    else:
        aug_negative_count = 0

    # padding zero if necessary
    if aug_positive_count > 0 and aug_negative_count > 0:
        aug_pred_bbox = torch.cat((aug_positive_rois, aug_negative_rois), dim=0)
        aug_deltas_pred_bbox = torch.cat(
            (aug_deltas_positive_pred, aug_deltas_negative_pred), dim=0
        )

        pred_logits_class = torch.cat(
            (positive_pred_logits_class, aug_pred_negative_logits_class), dim=0
        )
        pred_masks = torch.cat((positive_pred_masks, aug_pred_negative_masks), dim=0)

        zeros = (
            Variable(torch.zeros(aug_negative_count, 4), requires_grad=False)
            .int()
            .to(device)
        )
        aug_deltas_target_bbox = torch.cat([aug_deltas_target, zeros], dim=0)
        roi_target_bbox = torch.cat([aug_target_bbox, zeros], dim=0)

        zeros = (
            Variable(torch.zeros(aug_negative_count), requires_grad=False)
            .int()
            .to(device)
        )
        target_class = torch.cat([target_class, zeros])
        zeros = (
            Variable(torch.zeros(aug_negative_count, 768, 1280), requires_grad=False)
            .int()
            .to(device)
        )
        target_masks = torch.cat([target_masks, zeros])

    elif aug_positive_count > 0:
        aug_deltas_pred_bbox = aug_deltas_positive_pred
        aug_deltas_target_bbox = aug_deltas_target
        roi_target_bbox = aug_target_bbox
        pred_logits_class = positive_pred_logits_class
        pred_masks = positive_pred_masks

    elif aug_negative_count > 0:
        aug_deltas_pred_bbox = aug_deltas_negative_pred

        pred_logits_class = aug_pred_negative_logits_class
        pred_masks = aug_pred_negative_masks

        zeros = (
            Variable(torch.zeros(aug_negative_count, 4), requires_grad=False)
            .int()
            .to(device)
        )
        aug_deltas_target_bbox = torch.cat([aug_deltas_target, zeros], dim=0)
        roi_target_bbox = torch.cat([aug_target_bbox, zeros], dim=0)

        zeros = (
            Variable(torch.zeros(aug_negative_count), requires_grad=False)
            .int()
            .to(device)
        )
        target_class = torch.cat([target_class, zeros])
        zeros = (
            Variable(torch.zeros(aug_negative_count, 768, 1280), requires_grad=False)
            .int()
            .to(device)
        )
        target_masks = torch.cat([target_masks, zeros])
    else:
        aug_deltas_pred_bbox = Variable(torch.FloatTensor(), requires_grad=False).to(
            device
        )
        pred_logits_class = Variable(torch.LongTensor(), requires_grad=False).to(device)
        pred_masks = Variable(torch.LongTensor(), requires_grad=False).to(device)
        aug_deltas_target_bbox = Variable(torch.FloatTensor(), requires_grad=False).to(
            device
        )
        roi_target_bbox = Variable(torch.FloatTensor(), requires_grad=False).to(device)
        target_class = Variable(torch.LongTensor(), requires_grad=False).to(device)
        target_masks = Variable(torch.LongTensor(), requires_grad=False).to(device)

    return (
        aug_deltas_pred_bbox,
        pred_logits_class,
        pred_masks,
        aug_deltas_target_bbox,
        target_class,
        target_masks,
        roi_target_bbox,
    )


# Post prediction processing
def prediction(sample, model):
    input, target = sample

    input = [
        input["con_img"].to(device),
        input["img"].to(device),
        [input["boxes"].squeeze(0).to(device)],
        [input["track"].squeeze(0).to(device)],
    ]

    # forward
    model.eval().to(device)
    with torch.no_grad():
        results = model(input)

    # resize to original size
    boxes = [
        [
            (round(max(i[0] * 1728 / 1280, 0), 2), round(max(i[1] * 1296 / 768, 0), 2)),
            (round(max(i[2] * 1728 / 1280, 0), 2), round(max(i[3] * 1296 / 768, 0), 2)),
        ]
        for i in list(results[2]["boxes"].cpu().numpy())
    ]
    masks = results[2]["masks"]
    masks = [i for i in list(masks.detach().cpu().numpy())]
    cls = [str(i) for i in list(results[2]["labels"].int().cpu().numpy())]
    ids = [str(i) for i in list(results[2]["track"].int().cpu().numpy())]
    score = [round(i, 2) for i in list(results[2]["score"].cpu().numpy())]

    return boxes, cls, masks, ids, score


# applying mask to the image
def unmold_mask(mask, bbox, image_shape=data_params["FRAME_SIZE"]):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.2
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    mask1 = np.array(Image.fromarray(mask).resize((x2 - x1, y2 - y1))).astype(
        np.float32
    )
    mask2 = np.array(Image.fromarray(mask).resize(image_shape)).astype(np.float32)

    mask = np.where(mask1 >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape, dtype=np.uint8)

    try:
        full_mask[y1:y2, x1:x2] = mask
    except Exception as e:
        return full_mask

    return full_mask
