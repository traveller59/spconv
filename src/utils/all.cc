// Copyright 2019 Yan Yan
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <spconv/nms.h>
#include <spconv/point2voxel.h>
#include <spconv/box_iou.h>
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(spconv_utils, m)
{
    m.doc() = "util pybind11 functions for spconv";
    m.def("non_max_suppression", &spconv::non_max_suppression<double>, py::return_value_policy::reference_internal, "bbox iou",
          "boxes"_a = 1, "keep_out"_a = 2, "nms_overlap_thresh"_a = 3, "device_id"_a = 4);
    m.def("non_max_suppression", &spconv::non_max_suppression<float>, py::return_value_policy::reference_internal, "bbox iou",
          "boxes"_a = 1, "keep_out"_a = 2, "nms_overlap_thresh"_a = 3, "device_id"_a = 4);
    m.def("non_max_suppression_cpu", &spconv::non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou",
          "boxes"_a = 1, "order"_a = 2, "nms_overlap_thresh"_a = 3, "eps"_a = 4);
    m.def("non_max_suppression_cpu", &spconv::non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou",
          "boxes"_a = 1, "order"_a = 2, "nms_overlap_thresh"_a = 3, "eps"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &spconv::rotate_non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou",
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &spconv::rotate_non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou",
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
    m.def("rbbox_iou", &spconv::rbbox_iou<double>,
          py::return_value_policy::reference_internal, "rbbox iou",
          "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
          "standup_thresh"_a = 4);
    m.def("rbbox_iou", &spconv::rbbox_iou<float>,
          py::return_value_policy::reference_internal, "rbbox iou",
          "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
          "standup_thresh"_a = 4);
    m.def("points_to_voxel_3d_np", &spconv::points_to_voxel_3d_np<float, 3>,
          "matrix tensor_square", "points"_a = 1, "voxels"_a = 2, "coors"_a = 3,
          "num_points_per_voxel"_a = 4, "coor_to_voxelidx"_a = 5,
          "voxel_size"_a = 6, "coors_range"_a = 7, "max_points"_a = 8,
          "max_voxels"_a = 9);
    m.def("points_to_voxel_3d_np", &spconv::points_to_voxel_3d_np<double, 3>,
          "matrix tensor_square", "points"_a = 1, "voxels"_a = 2, "coors"_a = 3,
          "num_points_per_voxel"_a = 4, "coor_to_voxelidx"_a = 5,
          "voxel_size"_a = 6, "coors_range"_a = 7, "max_points"_a = 8,
          "max_voxels"_a = 9);
}