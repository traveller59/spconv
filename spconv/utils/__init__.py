# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from cumm import tensorview as tv
from contextlib import AbstractContextManager
from spconv.cppconstants import CPU_ONLY_BUILD
from spconv.core_cc.csrc.utils.boxops import BoxOps

from spconv.core_cc.csrc.sparse.all.ops_cpu1d import Point2VoxelCPU as Point2VoxelCPU1d
from spconv.core_cc.csrc.sparse.all.ops_cpu2d import Point2VoxelCPU as Point2VoxelCPU2d
from spconv.core_cc.csrc.sparse.all.ops_cpu3d import Point2VoxelCPU as Point2VoxelCPU3d
from spconv.core_cc.csrc.sparse.all.ops_cpu4d import Point2VoxelCPU as Point2VoxelCPU4d

if not CPU_ONLY_BUILD:
    from spconv.core_cc.csrc.sparse.all.ops1d import Point2Voxel as Point2VoxelGPU1d
    from spconv.core_cc.csrc.sparse.all.ops2d import Point2Voxel as Point2VoxelGPU2d
    from spconv.core_cc.csrc.sparse.all.ops3d import Point2Voxel as Point2VoxelGPU3d
    from spconv.core_cc.csrc.sparse.all.ops4d import Point2Voxel as Point2VoxelGPU4d


class nullcontext(AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def rbbox_iou(box_corners: np.ndarray, qbox_corners: np.ndarray,
              standup_iou: np.ndarray, standup_thresh: float):
    if not BoxOps.has_boost():
        raise NotImplementedError(
            "this op require spconv built with boost, download boost, export BOOST_ROOT and rebuild."
        )
    N = box_corners.shape[0]
    K = qbox_corners.shape[0]
    overlap = np.zeros((N, K), dtype=box_corners.dtype)

    BoxOps.rbbox_iou(tv.from_numpy(box_corners), tv.from_numpy(qbox_corners),
                     tv.from_numpy(standup_iou), tv.from_numpy(overlap),
                     standup_thresh, False)
    return overlap


def rbbox_intersection(box_corners: np.ndarray, qbox_corners: np.ndarray,
                       standup_iou: np.ndarray, standup_thresh: float):
    if not BoxOps.has_boost():
        raise NotImplementedError(
            "this op require spconv built with boost, download boost, export BOOST_ROOT and rebuild."
        )
    N = box_corners.shape[0]
    K = qbox_corners.shape[0]
    overlap = np.zeros((N, K), dtype=box_corners.dtype)

    BoxOps.rbbox_iou(tv.from_numpy(box_corners), tv.from_numpy(qbox_corners),
                     tv.from_numpy(standup_iou), tv.from_numpy(overlap),
                     standup_thresh, True)
    return overlap


def rbbox_iou_loss(box_corners: np.ndarray, qbox_corners: np.ndarray):
    if not BoxOps.has_boost():
        raise NotImplementedError(
            "this op require spconv built with boost, download boost, export BOOST_ROOT and rebuild."
        )
    N = box_corners.shape[0]
    overlap = np.zeros((N, ), dtype=box_corners.dtype)

    BoxOps.rbbox_iou_aligned(tv.from_numpy(box_corners),
                             tv.from_numpy(qbox_corners),
                             tv.from_numpy(overlap), False)
    return overlap


def non_max_suppression_cpu(boxes: np.ndarray,
                            order: np.ndarray,
                            thresh: float,
                            eps: float = 0.0):
    return BoxOps.non_max_suppression_cpu(tv.from_numpy(boxes),
                                          tv.from_numpy(order), thresh, eps)


def rotate_non_max_suppression_cpu(boxes: np.ndarray, order: np.ndarray,
                                   standup_iou: np.ndarray, thresh: float):
    if not BoxOps.has_boost():
        raise NotImplementedError(
            "this op require spconv built with boost, download boost, export BOOST_ROOT and rebuild."
        )
    return BoxOps.rotate_non_max_suppression_cpu(tv.from_numpy(boxes),
                                                 tv.from_numpy(order),
                                                 tv.from_numpy(standup_iou),
                                                 thresh)
