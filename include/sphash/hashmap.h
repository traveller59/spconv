#include <tensorview/tensor.h>


namespace spconv {

enum HashTypes {
    kDenseMap = 0,
    kCUDPPHash = 1
};

template <int Impl>
struct HashMap;

template<>
struct HashMap<kDenseMap>{

};

}