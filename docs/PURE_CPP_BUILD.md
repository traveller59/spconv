# Pure C++ build

Spconv provide a way to generate sources to build a C++ library with all op needed for inference and train.

The generated libspconv has been deployed to tensorrt and run in real automatic driving vehicles.

## Steps

1. Install spconv, Install cumm, Install cumm cmake, or add it as a sub directory

Cmake Install: clone cumm project, use ```mkdir -p build && cd build && cmake .. && make && make install```

Subdirectory: clone cumm project, copy it to your parent project.

2. prepare cmake list files, set some environment variables, then generate code

* Set Envs

```Bash
export CUMM_CUDA_VERSION=11.4 # cuda version, required
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
export CUMM_INCLUDE_PATH="\${CUMM_INCLUDE_PATH}" # if you use cumm as a subdirectory, you need this to find cumm includes.
export CUMM_CUDA_ARCH_LIST="6.1;7.5;8.6" # cuda arch flags
```

* Generate Code: Ignore train ops:

```Bash
python -m spconv.gencode --include=/path/to/spconv/include --src=/path/to/spconv/src --inference_only=True
```

* Generate Code: Include train ops: 

```Bash
python -m spconv.gencode --include=/path/to/spconv/include --src=/path/to/spconv/src
```

3. Run cmake build.


## Example

see example/libspconv/README.md
