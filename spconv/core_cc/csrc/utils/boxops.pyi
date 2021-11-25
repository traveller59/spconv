from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class BoxOps:
    @staticmethod
    def has_boost() -> bool: ...
    @staticmethod
    def non_max_suppression_cpu(boxes: Tensor, order: Tensor, thresh: float, eps: float = 0) -> List[int]: 
        """
        Args:
            boxes: 
            order: 
            thresh: 
            eps: 
        """
        ...
    @staticmethod
    def rotate_non_max_suppression_cpu(box_corners: Tensor, order: Tensor, standup_iou: Tensor, thresh: float, eps: float = 0) -> List[int]: 
        """
        Args:
            box_corners: 
            order: 
            standup_iou: 
            thresh: 
            eps: 
        """
        ...
    @staticmethod
    def rbbox_iou(box_corners: Tensor, qbox_corners: Tensor, standup_iou: Tensor, overlaps: Tensor, standup_thresh: float, inter_only: bool) -> None: 
        """
        Args:
            box_corners: 
            qbox_corners: 
            standup_iou: 
            overlaps: 
            standup_thresh: 
            inter_only: 
        """
        ...
    @staticmethod
    def rbbox_iou_aligned(box_corners: Tensor, qbox_corners: Tensor, overlaps: Tensor, inter_only: bool) -> None: 
        """
        Args:
            box_corners: 
            qbox_corners: 
            overlaps: 
            inter_only: 
        """
        ...
