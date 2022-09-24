# Copyright 2022 Yan Yan
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

from cumm.common import TensorView, TensorViewNVRTC

class PointCloudCompress(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, TensorViewNVRTC)
        self.add_include("unordered_map")
        self.add_include("tensorview/hash/hash_functions.h")
        self.add_enum_class("EncodeType", [
            ("XYZ_8", 0),
            ("XYZI_8", 1),
        ])

    @pccm.pybind.mark 
    @pccm.static_function
    def encode_with_order(self):
        code = pccm.code()
        # TODO add checksum
        code.arg("points", "tv::Tensor")
        code.arg("intensity", "tv::Tensor")

        code.arg("ex", "float")
        code.arg("ey", "float")
        code.arg("ez", "float")
        code.arg("type", "EncodeType")

        code.arg("with_order", "bool", "false")
        code.raw(f"""
        namespace op = tv::arrayops;
        float vx = 256.0f * ex;
        float vy = 256.0f * ey;
        float vz = 256.0f * ez;
        std::vector<std::tuple<uint64_t, tv::array<float, 3>>> offsets;
        std::unordered_map<uint64_t, std::vector<int64_t>> order;
        auto N = points.dim(0);
        tv::array<float, 3> vsize{{vx, vy, vz}};
        tv::array<float, 3> errors{{ex, ey, ez}};

        tv::Tensor order_ten;
        int64_t* order_ten_ptr = nullptr;

        if (with_order){{
            order_ten = tv::empty({{N}}, tv::int64);
            order_ten_ptr = order_ten.data_ptr<int64_t>();
        }}

        using hash_t = tv::hash::SpatialHash<uint64_t>;
        auto point_stride = points.stride(0);
        int64_t final_size = sizeof(int64_t) * 5 + sizeof(float) * 3;
        tv::Tensor res;
        tv::dispatch<float, double>(points.dtype(), [&](auto IP){{
            using TPoint = TV_DECLTYPE(IP);

            auto points_data = points.data_ptr<TPoint>();
            tv::dispatch<float, double, uint8_t>(intensity.dtype(), [&](auto II){{
                using TInten = TV_DECLTYPE(II); 
                auto intensity_data = intensity.data_ptr<TInten>();
                tv::dispatch_int<static_cast<int>(EncodeType::XYZI_8), static_cast<int>(EncodeType::XYZ_8)>(static_cast<int>(type), [&](auto I){{
                    constexpr int kTypeInt = TV_DECLTYPE(I)::value;
                    constexpr int kEncodeDim = kTypeInt == static_cast<int>(EncodeType::XYZI_8) ? 4 : 3;
                    std::unordered_map<uint64_t, std::vector<tv::array<uint8_t, kEncodeDim>>> hash;
                    int inten_stride = 0;
                    if (kEncodeDim > 3){{
                        TV_ASSERT_RT_ERR(!intensity.empty(), "inten must not empty");
                        inten_stride = intensity.stride(0);
                    }}

                    for (size_t i = 0; i < N; ++i){{
                        tv::array<float, 3> point = op::read_ptr<3>(points_data).template cast<float>();
                        auto pos_unit_voxel = point / vsize;
                        auto pos_int = op::apply(floorf, pos_unit_voxel).cast<int32_t>();
                        auto pos_enc = (point / errors - pos_int.cast<float>() * float(256)).cast<uint8_t>();
                        tv::array<uint8_t, kEncodeDim> enc;
                        enc[0] = pos_enc[0];
                        enc[1] = pos_enc[1];
                        enc[2] = pos_enc[2];
                        if (kEncodeDim > 3){{
                            TInten inten = intensity_data[0];
                            enc[3] = uint8_t(inten);
                        }}
                        auto pos_uint = pos_int + hash_t::direct_hash_offset();
                        uint64_t scalar = hash_t::encode(pos_int[0], pos_int[1], pos_int[2]);
                        auto iter = hash.find(scalar);
                        if (iter == hash.end()){{
                            auto pos_offset = pos_int.cast<float>() * vsize;
                            std::vector<tv::array<uint8_t, kEncodeDim>> vec{{enc}};
                            offsets.push_back({{scalar, pos_offset}});
                            hash.insert({{scalar, vec}});
                            final_size += sizeof(float) * 3 + sizeof(int) + sizeof(uint8_t) * kEncodeDim;
                            if (with_order){{
                                std::vector<int64_t> order_cluster{{int64_t(i)}};
                                order.insert({{scalar, order_cluster}});
                            }}
                        }}else{{
                            // iter.value().push_back(enc);
                            iter->second.push_back(enc);
                            final_size += sizeof(uint8_t) * kEncodeDim;
                            if (with_order){{
                                order.at(scalar).push_back(i);
                            }}
                        }}
                        points_data += point_stride;

                    }}
                    res = tv::empty({{final_size}}, tv::uint8, -1);
                    auto res_ptr = res.raw_data();
                    int64_t* res_ptr_header = reinterpret_cast<int64_t*>(res_ptr);
                    res_ptr_header[0] = int64_t(final_size); 
                    res_ptr_header[1] = static_cast<int>(type);
                    res_ptr_header[2] = int64_t(N);
                    res_ptr_header[3] = int64_t(offsets.size());
                    res_ptr_header[4] = 0;

                    // TODO add checksum in header
                    res_ptr += sizeof(int64_t) * 5;
                    float* error_header = reinterpret_cast<float*>(res_ptr);
                    error_header[0] = errors[0];
                    error_header[1] = errors[1];
                    error_header[2] = errors[2];
                    res_ptr += sizeof(float) * 3;
                    for (auto& p : offsets){{
                        auto& offset = std::get<1>(p);
                        auto& encodes = hash.at(std::get<0>(p));
                        int cluster_size = encodes.size();
                        reinterpret_cast<int*>(res_ptr)[0] = cluster_size;
                        reinterpret_cast<float*>(res_ptr)[1] = offset[0];
                        reinterpret_cast<float*>(res_ptr)[2] = offset[1];
                        reinterpret_cast<float*>(res_ptr)[3] = offset[2];
                        res_ptr += sizeof(int) + sizeof(float) * 3;
                    }}
                    for (auto& p : offsets){{
                        auto& offset = std::get<1>(p);
                        auto& encodes = hash.at(std::get<0>(p));
                        int cluster_size = encodes.size();
                        auto enc_ptr = reinterpret_cast<tv::array<uint8_t, kEncodeDim>*>(res_ptr);
                        for (int i = 0; i < cluster_size; ++i){{
                            enc_ptr[i] = encodes[i];
                        }}
                        if (with_order){{
                            auto& orders = order.at(std::get<0>(p));
                            for (int i = 0; i < cluster_size; ++i){{
                                order_ten_ptr[i] = orders[i];
                            }}
                            order_ten_ptr += cluster_size;
                        }}
                        res_ptr += cluster_size * sizeof(tv::array<uint8_t, kEncodeDim>);
                    }}
                    TV_ASSERT_RT_ERR(res_ptr - res.raw_data() == final_size, "error");
                }});
            }});
        }});
        return std::make_tuple(res, order_ten);
        """)
        return code.ret("std::tuple<tv::Tensor, tv::Tensor>")

    @pccm.pybind.mark 
    @pccm.static_function
    def encode_xyzi(self):
        code = pccm.code()
        code.arg("points", "tv::Tensor")
        code.arg("intensity", "tv::Tensor")

        code.arg("ex", "float")
        code.arg("ey", "float")
        code.arg("ez", "float")
        code.raw(f"""
        auto res = encode_with_order(points, intensity, ex, ey, ez, EncodeType::XYZI_8, false);
        return std::get<0>(res);
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark 
    @pccm.static_function
    def encode_xyz(self):
        code = pccm.code()
        code.arg("points", "tv::Tensor")
        code.arg("ex", "float")
        code.arg("ey", "float")
        code.arg("ez", "float")
        code.raw(f"""
        auto res = encode_with_order(points, tv::Tensor(), ex, ey, ez, EncodeType::XYZ_8, false);
        return std::get<0>(res);
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark 
    @pccm.static_function
    def decode(self):
        code = pccm.code()
        code.arg("data", "tv::Tensor")
        code.raw(f"""
        namespace op = tv::arrayops;
        const uint8_t* data_ptr = data.data_ptr<const uint8_t>();

        auto res_ptr = data.raw_data();
        int64_t* res_ptr_header = reinterpret_cast<int64_t*>(res_ptr);
        int64_t final_size = res_ptr_header[0];
        int type = res_ptr_header[1];
        int64_t N = res_ptr_header[2];
        int64_t voxel_num = res_ptr_header[3];

        TV_ASSERT_RT_ERR(final_size == data.raw_size(), "size mismatch");
        res_ptr += sizeof(int64_t) * 5;
        float* error_header = reinterpret_cast<float*>(res_ptr);
        tv::array<float, 3> error;
        error[0] = error_header[0];
        error[1] = error_header[1];
        error[2] = error_header[2];
        res_ptr += sizeof(float) * 3;
        tv::Tensor points;
        tv::dispatch_int<static_cast<int>(EncodeType::XYZI_8), static_cast<int>(EncodeType::XYZ_8)>(static_cast<int>(type), [&, error](auto I){{
            constexpr int kTypeInt = TV_DECLTYPE(I)::value;
            constexpr int kEncodeDim = kTypeInt == static_cast<int>(EncodeType::XYZI_8) ? 4 : 3;
            points = tv::empty({{N, kEncodeDim}}, tv::float32);
            auto points_ptr = points.data_ptr<float>();

            auto enc_ptr = reinterpret_cast<tv::array<uint8_t, kEncodeDim>*>(res_ptr + voxel_num * (sizeof(int) * 1 + sizeof(float) * 3));
            for (int i = 0; i < voxel_num; ++i){{
                int cluster_size = reinterpret_cast<int*>(res_ptr)[0];
                tv::array<float, 3> offset;
                offset[0] = reinterpret_cast<float*>(res_ptr)[1];
                offset[1] = reinterpret_cast<float*>(res_ptr)[2];
                offset[2] = reinterpret_cast<float*>(res_ptr)[3];
                auto point_cur_ptr = points_ptr;
                for (int j = 0; j < cluster_size; ++j){{
                    auto& enc = enc_ptr[j];
                    tv::array<float, 3> point = op::slice<0, 3>(enc).template cast<float>() * error + offset;
                    point_cur_ptr[0] = point[0];
                    point_cur_ptr[1] = point[1];
                    point_cur_ptr[2] = point[2];
                    if (kEncodeDim > 3){{
                        point_cur_ptr[3] = enc[3];
                    }}
                    point_cur_ptr += kEncodeDim;
                }}
                res_ptr += sizeof(int) + sizeof(float) * 3;
                enc_ptr += cluster_size;
                points_ptr += cluster_size * kEncodeDim;
            }}
        }});
        return points;
        """)
        return code.ret("tv::Tensor")

