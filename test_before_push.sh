# developers must run this file before push or pull request.
# this script contains three parts:
# 1. unit tests for all gemm/conv kernels
# 2. comparison test: compare network fwd/bwd results between CPU, Native, ImplicitGemm
# 3. f32/f16 train/eval test based on mnist and some small datasets

echo "-------------UNIT TEST START--------------"
pytest ./test 
echo "-------------UNIT TEST END--------------"
python ./example/mnist_sparse.py --fp16