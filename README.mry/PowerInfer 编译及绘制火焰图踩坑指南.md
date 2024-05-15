# PowerInfer CPU

## 1 编译

- Release 版本

  cmake -S . -B build
  cmake --build build --config Release

- Debug 版本

  cmake -S . -DCMAKE_BUILD_TYPE=Debug -B build

  cmake --build build --config Debug

## 2 火焰图

- 编译Debug版本

- git clone https://github.com/brendangregg/FlameGraph.git

- perf record

  - 线程数=1

    sudo perf record -a -g ./build/bin/main -m ./ReluLLaMA-7B/llama-7b-relu.q4.powerinfer.gguf -n 128 -t 1 -p "Once upon a time, there existed a littel girl,"

  - 线程数=8

    sudo perf record -a -g ./build/bin/main -m ./ReluLLaMA-7B/llama-7b-relu.q4.powerinfer.gguf -n 128 -t 8 -p "Once upon a time, there existed a littel girl,"

- sudo perf script -i perf.data &> perf.unfold

- sudo FlameGraph/stackcollapse-perf.pl perf.unfold &> perf.folded

- sudo FlameGraph/flamegraph.pl perf.folded > perf.svg