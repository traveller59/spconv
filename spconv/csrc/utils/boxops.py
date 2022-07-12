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

import pccm 
from pathlib import Path 
import os 
from cumm.common import TensorView, TensorViewCPU, TensorViewKernel, ThrustLib
from spconv.constants import BOOST_ROOT


class BoostGeometryLib(pccm.Class):
    def __init__(self):
        super().__init__()
        assert BOOST_ROOT is not None 
        self.build_meta.add_public_includes(BOOST_ROOT)
        self.add_include("boost/geometry.hpp")

class BoxOps(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
    
    @pccm.pybind.mark
    @pccm.static_function
    def has_boost(self):
        code = pccm.FunctionCode()
        code.raw(f"return {pccm.boolean(BOOST_ROOT is not None)};")
        return code.ret("bool")

    @pccm.pybind.mark(nogil=True)
    @pccm.static_function
    def non_max_suppression_cpu(self):
        code = pccm.FunctionCode()
        code.arg("boxes, order", "tv::Tensor")
        code.arg("thresh", "float")
        code.arg("eps", "float", "0")

        code.raw(f"""
        auto ndets = boxes.dim(0);
        std::vector<int> keep(ndets);

        tv::dispatch<float, double>(boxes.dtype(), [&](auto I1){{
            using DType = TV_DECLTYPE(I1);
            auto boxes_r = boxes.tview<const DType, 2>();
            tv::dispatch<int, int64_t, uint32_t, uint64_t>(order.dtype(), [&](auto I2){{
                using T2 = TV_DECLTYPE(I2);
                auto order_r = order.tview<const T2, 1>();
                std::vector<DType> areas;
                for (int i = 0; i < ndets; ++i){{
                    areas[i] = (boxes_r(i, 2) - boxes_r(i, 0) + eps) * 
                               (boxes_r(i, 3) - boxes_r(i, 1) + eps);
                }}
                std::vector<int> suppressed(ndets, 0);

                int i, j;
                DType xx1, xx2, w, h, inter, ovr;
                for (int _i = 0; _i < ndets; ++_i) {{
                    i = order_r(_i);
                    if (suppressed[i] == 1)
                        continue;
                    keep.push_back(i);
                    for (int _j = _i + 1; _j < ndets; ++_j) {{
                        j = order_r(_j);
                        if (suppressed[j] == 1)
                            continue;
                        xx2 = std::min(boxes_r(i, 2), boxes_r(j, 2));
                        xx1 = std::max(boxes_r(i, 0), boxes_r(j, 0));
                        w = xx2 - xx1 + eps;
                        if (w > 0) {{
                            xx2 = std::min(boxes_r(i, 3), boxes_r(j, 3));
                            xx1 = std::max(boxes_r(i, 1), boxes_r(j, 1));
                            h = xx2 - xx1 + eps;
                            if (h > 0) {{
                            inter = w * h;
                            ovr = inter / (areas[i] + areas[j] - inter);
                            if (ovr >= thresh)
                                suppressed[j] = 1;
                            }}
                        }}
                    }}
                }}
            }});

        }});
        return keep;
        """)
        return code.ret("std::vector<int>")

    @pccm.pybind.mark(nogil=True)
    @pccm.static_function
    def rotate_non_max_suppression_cpu(self):
        code = pccm.FunctionCode()
        code.arg("box_corners, order, standup_iou", "tv::Tensor")
        code.arg("thresh", "float")
        code.arg("eps", "float", "0")
        if BOOST_ROOT is None:
            return code.make_invalid()
        code.add_dependency(BoostGeometryLib)
        code.raw(f"""
        auto ndets = box_corners.dim(0);
        std::vector<int> keep(ndets);

        tv::dispatch<float, double>(box_corners.dtype(), [&](auto I1){{
            using DType = TV_DECLTYPE(I1);
            auto box_corners_r = box_corners.tview<const DType, 3>();
            auto standup_iou_r = standup_iou.tview<const DType, 2>();

            tv::dispatch<int, int64_t, uint32_t, uint64_t>(order.dtype(), [&](auto I2){{
                using T2 = TV_DECLTYPE(I2);
                auto order_r = order.tview<const T2, 1>();
                std::vector<int> suppressed(ndets, 0);
                int i, j;

                namespace bg = boost::geometry;
                typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
                typedef bg::model::polygon<point_t> polygon_t;
                polygon_t poly, qpoly;
                std::vector<polygon_t> poly_inter, poly_union;
                DType inter_area, union_area, overlap;

                for (int _i = 0; _i < ndets; ++_i) {{
                    i = order_r(_i);
                    if (suppressed[i] == 1)
                    continue;
                    keep.push_back(i);
                    for (int _j = _i + 1; _j < ndets; ++_j) {{
                        j = order_r(_j);
                        if (suppressed[j] == 1)
                            continue;
                        if (standup_iou_r(i, j) <= 0.0)
                            continue;
                        // std::cout << "pre_poly" << std::endl;
                        bg::append(poly,
                                point_t(box_corners_r(i, 0, 0), box_corners_r(i, 0, 1)));
                        bg::append(poly,
                                point_t(box_corners_r(i, 1, 0), box_corners_r(i, 1, 1)));
                        bg::append(poly,
                                point_t(box_corners_r(i, 2, 0), box_corners_r(i, 2, 1)));
                        bg::append(poly,
                                point_t(box_corners_r(i, 3, 0), box_corners_r(i, 3, 1)));
                        bg::append(poly,
                                point_t(box_corners_r(i, 0, 0), box_corners_r(i, 0, 1)));
                        bg::append(qpoly,
                                point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
                        bg::append(qpoly,
                                point_t(box_corners_r(j, 1, 0), box_corners_r(j, 1, 1)));
                        bg::append(qpoly,
                                point_t(box_corners_r(j, 2, 0), box_corners_r(j, 2, 1)));
                        bg::append(qpoly,
                                point_t(box_corners_r(j, 3, 0), box_corners_r(j, 3, 1)));
                        bg::append(qpoly,
                                point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
                        bg::intersection(poly, qpoly, poly_inter);
                        if (!poly_inter.empty()) {{
                            inter_area = bg::area(poly_inter.front());
                            bg::union_(poly, qpoly, poly_union);
                            if (!poly_union.empty()) {{ // ignore invalid box
                                union_area = bg::area(poly_union.front());
                                overlap = inter_area / union_area;
                                if (overlap >= thresh)
                                    suppressed[j] = 1;
                                poly_union.clear();
                            }}
                        }}
                        poly.clear();
                        qpoly.clear();
                        poly_inter.clear();
                    }}
                }}
            }});
        }});
        return keep;
        """)
        return code.ret("std::vector<int>")

    @pccm.pybind.mark(nogil=True)
    @pccm.static_function
    def rbbox_iou(self):
        code = pccm.FunctionCode()
        code.arg("box_corners, qbox_corners, standup_iou, overlaps", "tv::Tensor")
        code.arg("standup_thresh", "float")
        code.arg("inter_only", "bool")

        if BOOST_ROOT is None:
            return code.make_invalid()
        code.add_dependency(BoostGeometryLib)
        code.raw(f"""
        auto N = box_corners.dim(0);
        auto K = qbox_corners.dim(0);
        if (N == 0 || K == 0) {{
            return;
        }}
        tv::dispatch<float, double>(box_corners.dtype(), [&](auto I1){{
            using DType = TV_DECLTYPE(I1);

            auto box_corners_r = box_corners.tview<const DType, 3>();
            auto qbox_corners_r = qbox_corners.tview<const DType, 3>();

            auto standup_iou_r = standup_iou.tview<const DType, 2>();
            auto overlaps_rw = overlaps.tview<DType, 2>();

            namespace bg = boost::geometry;
            typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
            typedef bg::model::polygon<point_t> polygon_t;
            polygon_t poly, qpoly;
            std::vector<polygon_t> poly_inter, poly_union;
            DType inter_area, union_area;
            for (int k = 0; k < K; ++k) {{
                for (int n = 0; n < N; ++n) {{
                    if (standup_iou_r(n, k) <= standup_thresh)
                        continue;
                    bg::append(poly, point_t(box_corners_r(n, 0, 0), box_corners_r(n, 0, 1)));
                    bg::append(poly, point_t(box_corners_r(n, 1, 0), box_corners_r(n, 1, 1)));
                    bg::append(poly, point_t(box_corners_r(n, 2, 0), box_corners_r(n, 2, 1)));
                    bg::append(poly, point_t(box_corners_r(n, 3, 0), box_corners_r(n, 3, 1)));
                    bg::append(poly, point_t(box_corners_r(n, 0, 0), box_corners_r(n, 0, 1)));
                    bg::append(qpoly,
                                point_t(qbox_corners_r(k, 0, 0), qbox_corners_r(k, 0, 1)));
                    bg::append(qpoly,
                                point_t(qbox_corners_r(k, 1, 0), qbox_corners_r(k, 1, 1)));
                    bg::append(qpoly,
                                point_t(qbox_corners_r(k, 2, 0), qbox_corners_r(k, 2, 1)));
                    bg::append(qpoly,
                                point_t(qbox_corners_r(k, 3, 0), qbox_corners_r(k, 3, 1)));
                    bg::append(qpoly,
                                point_t(qbox_corners_r(k, 0, 0), qbox_corners_r(k, 0, 1)));

                    bg::intersection(poly, qpoly, poly_inter);

                    if (!poly_inter.empty()) {{
                        inter_area = bg::area(poly_inter.front());
                        if (inter_only){{
                            overlaps_rw(n, k) = inter_area;
                        }}else{{
                            bg::union_(poly, qpoly, poly_union);
                            if (!poly_union.empty()) {{
                                union_area = bg::area(poly_union.front());
                                overlaps_rw(n, k) = inter_area / union_area;
                            }}
                            poly_union.clear();
                        }}
                    }}
                    poly.clear();
                    qpoly.clear();
                    poly_inter.clear();
                }}
            }}
        }});
        return;
        """)
        return code

    @pccm.pybind.mark(nogil=True)
    @pccm.static_function
    def rbbox_iou_aligned(self):
        code = pccm.FunctionCode()
        code.arg("box_corners, qbox_corners, overlaps", "tv::Tensor")
        code.arg("inter_only", "bool")

        if BOOST_ROOT is None:
            return code.make_invalid()
        code.add_dependency(BoostGeometryLib)
        code.raw(f"""
        auto N = box_corners.dim(0);
        auto K = qbox_corners.dim(0);
        TV_ASSERT_RT_ERR(N == K, "aligned iou must have same number of box")
        if (N == 0 || K == 0) {{
            return;
        }}
        tv::dispatch<float, double>(box_corners.dtype(), [&](auto I1){{
            using DType = TV_DECLTYPE(I1);

            auto box_corners_r = box_corners.tview<const DType, 3>();
            auto qbox_corners_r = qbox_corners.tview<const DType, 3>();

            auto overlaps_rw = overlaps.tview<DType, 1>();

            namespace bg = boost::geometry;
            typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
            typedef bg::model::polygon<point_t> polygon_t;
            polygon_t poly, qpoly;
            std::vector<polygon_t> poly_inter, poly_union;
            DType inter_area, union_area;

            for (int n = 0; n < N; ++n) {{
                bg::append(poly, point_t(box_corners_r(n, 0, 0), box_corners_r(n, 0, 1)));
                bg::append(poly, point_t(box_corners_r(n, 1, 0), box_corners_r(n, 1, 1)));
                bg::append(poly, point_t(box_corners_r(n, 2, 0), box_corners_r(n, 2, 1)));
                bg::append(poly, point_t(box_corners_r(n, 3, 0), box_corners_r(n, 3, 1)));
                bg::append(poly, point_t(box_corners_r(n, 0, 0), box_corners_r(n, 0, 1)));
                bg::append(qpoly,
                            point_t(qbox_corners_r(n, 0, 0), qbox_corners_r(n, 0, 1)));
                bg::append(qpoly,
                            point_t(qbox_corners_r(n, 1, 0), qbox_corners_r(n, 1, 1)));
                bg::append(qpoly,
                            point_t(qbox_corners_r(n, 2, 0), qbox_corners_r(n, 2, 1)));
                bg::append(qpoly,
                            point_t(qbox_corners_r(n, 3, 0), qbox_corners_r(n, 3, 1)));
                bg::append(qpoly,
                            point_t(qbox_corners_r(n, 0, 0), qbox_corners_r(n, 0, 1)));

                bg::intersection(poly, qpoly, poly_inter);

                if (!poly_inter.empty()) {{
                    inter_area = bg::area(poly_inter.front());
                    if (inter_only){{
                        overlaps_rw(n) = inter_area;
                    }}else{{
                        bg::union_(poly, qpoly, poly_union);
                        if (!poly_union.empty()) {{
                            union_area = bg::area(poly_union.front());
                            overlaps_rw(n) = inter_area / union_area;
                        }}
                        poly_union.clear();
                    }}
                }}
                poly.clear();
                qpoly.clear();
                poly_inter.clear();
            }}
        }});
        return;
        """)
        return code
