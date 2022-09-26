# remove previous cloned cumm first.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

git clone https://github.com/FindDefinition/cumm.git $SCRIPT_DIR/cumm

export CUMM_CUDA_VERSION=11.4 # cuda version, required but only used for flag selection when build libspconv.
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
export CUMM_INCLUDE_PATH="\${CUMM_INCLUDE_PATH}" # if you use cumm as a subdirectory, you need this to find cumm includes.
export CUMM_CUDA_ARCH_LIST="7.5;8.6" # cuda arch flags

python -m spconv.gencode --include=$SCRIPT_DIR/spconv/include --src=$SCRIPT_DIR/spconv/src --inference_only=True


mkdir -p $SCRIPT_DIR/build
cd $SCRIPT_DIR/build
cmake ..
cmake --build $SCRIPT_DIR/build --config Release -j 8 # --verbose
