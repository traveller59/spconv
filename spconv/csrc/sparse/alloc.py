import pccm 
from cumm.common import TensorView, TensorViewCPU, TensorViewKernel, ThrustLib

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
        return code.ret("tv::Tensor")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True, pure_virtual=True)
    def empty(self):
        code = pccm.code()
        code.arg("name", "std::string")
        code.arg("shape", "std::vector<int64_t>")
        code.arg("dtype", "int")
        code.arg("device", "int")
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
        code.raw(f"""
        // "" means temp memory
        auto ten = zeros("", shape, dtype, device);
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
        code.raw(f"""
        auto ten = empty("", shape, dtype, device);
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
        code.raw(f"""
        auto ten = full_int("", shape, value, dtype, device);
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
        code.raw(f"""
        auto ten = full_float("", shape, value, dtype, device);
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
        auto ten = allocator_.empty("", {{num_bytes}}, tv::uint8, 0);
        return reinterpret_cast<char*>(ten.raw_data());
        """)
        return code

    @pccm.member_function
    def deallocate(self):
        code = pccm.FunctionCode()
        code.arg("ptr", "char *")
        code.arg("num_bytes", "size_t")
        code.raw(f"""
        return allocator_.free_noexcept(tv::from_blob(ptr, {{num_bytes}}, tv::uint8, 0));
        """)
        return code        
