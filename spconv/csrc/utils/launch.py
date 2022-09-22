import pccm
from cumm.common import TensorView

class LaunchUtils(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("limits")
        self.add_dependency(TensorView)
        self.add_static_const("kMaxGridYZDim", "int", "65535")

    @pccm.static_function
    def get_blocks_threads_of_2d_tensor(self):
        code = pccm.code()
        code.arg("nhot", "int64_t")
        code.arg("num_features", "int64_t")

        code.raw(f"""
        constexpr int MaxThreads = 512;
        int num_blocks_X = 0;
        int num_blocks_Y = 0;
        int threads_X = 0;
        int threads_Y = 0;

        dim3 threads;
        bool found = tv::dispatch_int_noexcept<512, 256, 128, 64, 32, 16>(int(num_features), [](int my, int expect){{return my >= expect;}}, [&](auto V){{
            // if num_features > value in list above, run this function.
            // if a value is found, other value won't be executed.
            int NumFeatures = TV_DECLTYPE(V)::value;
            int Num0 = MaxThreads / NumFeatures;
            num_blocks_X = tv::div_up(num_features, int64_t(NumFeatures));
            num_blocks_Y = tv::div_up(nhot, int64_t(Num0));
            threads_X = NumFeatures;
            threads_Y = Num0;
        }});
        if (!found){{
            int NumFeatures = 16;
            int Num0 = MaxThreads / NumFeatures;
            num_blocks_X = tv::div_up(num_features, int64_t(NumFeatures));
            num_blocks_Y = tv::div_up(nhot, int64_t(Num0));
            threads_X = NumFeatures;
            threads_Y = Num0;
        }}
        return std::make_tuple(num_blocks_X, num_blocks_Y, threads_X, threads_Y);
        """)
        code.ret("std::tuple<int, int, int, int>")
        return code 

