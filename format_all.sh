isort -rc --atomic ./spconv && \
isort -rc --atomic ./test && \
yapf -i --recursive -vv ./spconv ./test
find ./src -regex '.*\.\(cpp\|hpp\|cc\|cxx\|cu\|cuh\|h\)' | xargs clang-format -i
find ./include -regex '.*\.\(cpp\|hpp\|cc\|cxx\|cu\|cuh\|h\)' | xargs clang-format -i