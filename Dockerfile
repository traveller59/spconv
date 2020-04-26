FROM scrin/dev:latest

RUN PROBLEM_FILE=/usr/local/lib/python3.8/dist-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake && \
    sed -i 's/-Wall;-Wextra;-Wno-unused-parameter;-Wno-missing-field-initializers;-Wno-write-strings;-Wno-unknown-pragmas;-Wno-missing-braces;-fopenmp//g' $PROBLEM_FILE && \
    sed -i 's/-Wall;-Wextra;-Wno-unused-parameter;-Wno-missing-field-initializers;-Wno-write-strings;-Wno-unknown-pragmas;-Wno-missing-braces//g' $PROBLEM_FILE && \
    cd /root && \
    git clone --depth 1 --recursive https://www.github.com/traveller59/spconv.git && \
    cd ./spconv && \
    SPCONV_FORCE_BUILD_CUDA=1 python setup.py install
