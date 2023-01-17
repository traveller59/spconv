import pccm 
from cumm.common import TensorView, TensorViewCPU, TensorViewKernel, ThrustLib

from spconv.constants import AllocKeys
from cumm.constants import CUMM_CPU_ONLY_BUILD
from .indices import CudaCommonKernel
class ExternalAllocatorGuard(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_member("tensor", "tv::Tensor")
        self.add_member("free_func", "std::function<void(tv::Tensor)>")

    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("ten", "tv::Tensor")
        code.arg("free_func", "std::function<void(tv::Tensor)>")
        code.ctor_init("tensor", "ten")
        code.ctor_init("free_func", "free_func")
        return code 

    @pccm.constructor
    def dctor(self):
        code = pccm.code()
        return code 

    @pccm.destructor
    def dtor(self):
        code = pccm.code()
        code.raw(f"""
        if (!tensor.empty() && free_func){{
            free_func(tensor);
        }}
        """)
        return code

class ExternalAllocator(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, ExternalAllocatorGuard)
        self.use_shared = True
        self.ptr_type = "unique"
        if self.use_shared:
            self.ptr_type = "shared"

        self.add_typedef("guard_t", f"std::{self.ptr_type}_ptr<ExternalAllocatorGuard>")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True, pure_virtual=True)
    def zeros(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("stream", "std::uintptr_t", "0")
        code.arg("is_temp_memory", "bool", "false")
        code.arg("scale", "float", "1.0")
        return code.ret("tv::Tensor")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True, pure_virtual=True)
    def empty(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("stream", "std::uintptr_t", "0")
        code.arg("is_temp_memory", "bool", "false")
        code.arg("scale", "float", "1.0")
        return code.ret("tv::Tensor")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True, pure_virtual=True)
    def full_int(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("value", "int")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("stream", "std::uintptr_t", "0")
        code.arg("is_temp_memory", "bool", "false")

        return code.ret("tv::Tensor")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True, pure_virtual=True)
    def full_float(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("value", "float")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("stream", "std::uintptr_t", "0")
        code.arg("is_temp_memory", "bool", "false")

        return code.ret("tv::Tensor")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True, pure_virtual=True)
    def get_tensor_by_name(self):
        code = pccm.code()
        code.arg("name", "std::string")
        return code.ret("tv::Tensor")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True, pure_virtual=True)
    def free(self):
        code = pccm.code()
        code.arg("ten", "tv::Tensor")
        return code

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True, pure_virtual=True)
    def free_noexcept(self):
        code = pccm.code()
        code.arg("ten", "tv::Tensor")
        return code

    @pccm.member_function
    def zeros_guard(self):
        code = pccm.code()
        code.arg("shape", "std::vector<int64_t>")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("name", "std::string", "\"\"")
        code.arg("stream", "std::uintptr_t", "0")
        code.raw(f"""
        // "" means temp memory
        auto ten = zeros(name, shape, dtype, device, stream, true);
        return std::make_{self.ptr_type}<ExternalAllocatorGuard>(ten, [this](tv::Tensor ten){{
            this->free(ten);
        }});
        """)
        return code.ret(f"std::{self.ptr_type}_ptr<ExternalAllocatorGuard>")

    @pccm.member_function
    def empty_guard(self):
        code = pccm.code()
        code.arg("shape", "std::vector<int64_t>")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("name", "std::string", "\"\"")
        code.arg("stream", "std::uintptr_t", "0")
        code.raw(f"""
        auto ten = empty(name, shape, dtype, device, stream, true);
        return std::make_{self.ptr_type}<ExternalAllocatorGuard>(ten, [this](tv::Tensor ten){{
            this->free(ten);
        }});
        """)
        return code.ret(f"std::{self.ptr_type}_ptr<ExternalAllocatorGuard>")

    @pccm.member_function
    def full_int_guard(self):
        code = pccm.code()
        code.arg("shape", "std::vector<int64_t>")
        code.arg("value", "int")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("name", "std::string", "\"\"")
        code.arg("stream", "std::uintptr_t", "0")
        code.raw(f"""
        auto ten = full_int(name, shape, value, dtype, device, stream, true);
        return std::make_{self.ptr_type}<ExternalAllocatorGuard>(ten, [this](tv::Tensor ten){{
            this->free(ten);
        }});
        """)
        return code.ret(f"std::{self.ptr_type}_ptr<ExternalAllocatorGuard>")
    
    @pccm.member_function
    def full_float_guard(self):
        code = pccm.code()
        code.arg("shape", "std::vector<int64_t>")
        code.arg("value", "int")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("name", "std::string", "\"\"")
        code.arg("stream", "std::uintptr_t", "0")
        code.raw(f"""
        auto ten = full_float(name, shape, value, dtype, device, stream, true);
        return std::make_{self.ptr_type}<ExternalAllocatorGuard>(ten, [this](tv::Tensor t){{
            this->free(t);
        }});
        """)
        return code.ret(f"std::{self.ptr_type}_ptr<ExternalAllocatorGuard>")
    
class ThrustAllocator(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, ExternalAllocator)
        self.add_include("functional", "memory")
        self.add_member("allocator_", "ExternalAllocator&",)
        self.add_typedef("value_type", "char")

    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("allocator", "ExternalAllocator&")
        code.ctor_init("allocator_", "allocator")
        return code 

    @pccm.member_function
    def allocate(self):
        code = pccm.FunctionCode()
        code.arg("num_bytes", "std::ptrdiff_t")
        code.ret("char*")
        code.raw(f"""
        auto ten = allocator_.empty({pccm.literal(AllocKeys.ThrustTemp)}, {{num_bytes}}, tv::uint8, 0);
        return reinterpret_cast<char*>(ten.raw_data());
        """)
        return code

    @pccm.member_function
    def deallocate(self):
        code = pccm.FunctionCode()
        code.arg("ptr", "char *")
        code.arg("num_bytes", "size_t")
        code.raw(f"""
        return allocator_.free_noexcept(tv::from_blob(ptr, {{int64_t(num_bytes)}}, tv::uint8, 0));
        """)
        return code



class StaticAllocator(ExternalAllocator):
    """a static allocator for tensorrt plugin.
    """
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_member("tensor_dict_", "std::unordered_map<std::string, tv::Tensor>")
        self.add_member("repr_", "std::string")
        self.add_member("thrust_tmp_tensor_", "tv::Tensor")
        self.grow = 1.5
        self.cuda_common_kernel = CudaCommonKernel()

    @pccm.pybind.mark 
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("tensor_dict", "std::unordered_map<std::string, tv::Tensor>")
        code.ctor_init("tensor_dict_", "tensor_dict")
        code.raw(f"""
        std::stringstream ss;
        for (auto& p : tensor_dict){{
            tv::sstream_print(ss, p.first, p.second.shape(), tv::dtype_str(p.second.dtype()), "\\n");
        }}
        repr_ = ss.str();
        """)
        return code 

    @pccm.pybind.mark 
    @pccm.member_function
    def set_new_tensor_dict(self):
        code = pccm.code()
        code.arg("tensor_dict", "std::unordered_map<std::string, tv::Tensor>")
        code.raw(f"""
        tensor_dict_ = tensor_dict;
        std::stringstream ss;
        for (auto& p : tensor_dict){{
            tv::sstream_print(ss, p.first, p.second.shape(), tv::dtype_str(p.second.dtype()), "\\n");
        }}
        repr_ = ss.str();
        """)
        return code 

    @pccm.member_function(virtual=True)
    def _get_raw_and_check(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("is_temp_memory", "bool", "false")

        code.raw(f"""
        auto res = get_tensor_by_name(name);
        size_t total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
        TV_ASSERT_RT_ERR(res.nbytes() >= total * tv::bit_size(tv::DType(dtype)) / 8 
            && res.device() == device, "alloc failed, tensor size too small", shape, res.shape());

        // if (is_temp_memory){{
        // }}else{{
        //     // size must exactly match
        //     TV_ASSERT_RT_ERR(res.nbytes() == total * tv::bit_size(tv::DType(dtype)) / 8 
        //         && res.device() == device, "alloc failed, named memory size must match", shape, res.shape());
        // }}
        return tv::from_blob(res.raw_data(), shape, tv::DType(dtype), device);
        """)
        return code.ret("tv::Tensor")


    @pccm.pybind.mark
    @pccm.member_function(virtual=True)
    def zeros(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("stream", "std::uintptr_t", "0")
        code.arg("is_temp_memory", "bool", "false")
        code.arg("scale", "float", "1.0")
        code.raw(f"""
        auto tvctx = tv::Context();
        """)
        if not CUMM_CPU_ONLY_BUILD:
            code.raw(f"""
            tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream));
            """)
        code.raw(f"""
        auto blob = _get_raw_and_check(name, shape, dtype, device, is_temp_memory);
        return blob.zero_(tvctx);
        """)
        return code.ret("tv::Tensor")


    @pccm.pybind.mark
    @pccm.member_function(virtual=True)
    def empty(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("stream", "std::uintptr_t", "0")
        code.arg("is_temp_memory", "bool", "false")
        code.arg("scale", "float", "1.0")
        code.raw(f"""
        if (name == {pccm.literal(AllocKeys.ThrustTemp)}){{
            // thrust tmp shouldn't inside tensor_dict. use a simple method to allocate
            // we assume each allocator always handle one stream
            // so we can just use one tensor
            tv::Tensor res = thrust_tmp_tensor_;
            if (res.empty()){{
                res = tv::empty(shape, tv::DType(dtype), device);
                thrust_tmp_tensor_ = res;
            }}
            if (shape[0] > thrust_tmp_tensor_.dim(0)){{
                res = tv::empty({{int64_t(shape[0] * {self.grow})}}, tv::DType(dtype), device);
                thrust_tmp_tensor_ = res;
            }}
            return res;
        }}else{{
            auto blob = _get_raw_and_check(name, shape, dtype, device, is_temp_memory);
            return blob;
        }}
        """)
        return code.ret("tv::Tensor")

    # cpu only build can't use pccm.cuda
    __CUDA_DECORATOR = pccm.member_function
    if not CUMM_CPU_ONLY_BUILD:
        __CUDA_DECORATOR = pccm.cuda.member_function

    @pccm.pybind.mark
    @__CUDA_DECORATOR
    def full_int(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("value", "int")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("stream", "std::uintptr_t", "0")
        code.arg("is_temp_memory", "bool", "false")

        code.raw(f"""
        auto tvctx = tv::Context();
        auto blob = _get_raw_and_check(name, shape, dtype, device, is_temp_memory);

        """)
        if not CUMM_CPU_ONLY_BUILD:
            code.add_param_class("cudakers", self.cuda_common_kernel)
            code.raw(f"""
            tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream));
            using ints_t = std::tuple<int32_t, int16_t, int8_t, int64_t, uint32_t, uint64_t, uint16_t, uint8_t>;
            tv::Dispatch<ints_t>()(blob.dtype(), [&](auto I){{
                using T = TV_DECLTYPE(I);
                tv::cuda::Launch lanucher_fill(blob.size(), reinterpret_cast<cudaStream_t>(stream));
                lanucher_fill(cudakers::fill_kernel<T>, blob.data_ptr<T>(), value, blob.size());
            }});
            """)
        else:
            code.raw(f"""
            blob.fill_(value);
            """)
        code.raw(f"""
        return blob;
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark
    @__CUDA_DECORATOR
    def full_float(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("value", "float")
        code.arg("dtype", "int")
        code.arg("device", "int")
        code.arg("stream", "std::uintptr_t", "0")
        code.arg("is_temp_memory", "bool", "false")
        code.raw(f"""
        auto tvctx = tv::Context();
        auto blob = _get_raw_and_check(name, shape, dtype, device, is_temp_memory);
        """)
        if not CUMM_CPU_ONLY_BUILD:
            code.add_param_class("cudakers", self.cuda_common_kernel)
            code.raw(f"""
            tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream));
            using dtypes_t = std::tuple<float, double>;
            tv::Dispatch<dtypes_t>()(blob.dtype(), [&](auto I){{
                using T = TV_DECLTYPE(I);
                tv::cuda::Launch lanucher_fill(blob.size(), reinterpret_cast<cudaStream_t>(stream));
                lanucher_fill(cudakers::fill_kernel<T>, blob.data_ptr<T>(), value, blob.size());
            }});
            """)
        else:
            code.raw(f"""
            blob.fill_(value);
            """)
        code.raw(f"""
        return blob;
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark
    @pccm.member_function(virtual=True)
    def get_tensor_by_name(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.raw(f"""
        TV_ASSERT_RT_ERR(tensor_dict_.find(name) != tensor_dict_.end(), "can't find", name, "exists:\\n", repr_);
        return tensor_dict_.at(name);
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark
    @pccm.member_function(virtual=True)
    def free(self):
        # nothing here because this is a static allocator
        code = pccm.code()
        code.arg("ten", "tv::Tensor")
        return code

    @pccm.pybind.mark
    @pccm.member_function(virtual=True)
    def free_noexcept(self):
        code = pccm.code()
        code.arg("ten", "tv::Tensor")
        return code


