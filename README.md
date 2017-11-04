# Autonomous Exploration with Expectation-Maximization

Random Environment         |  Structured Environment
:-------------------------:|:-------------------------:
![](./figures/isrr2017_random.gif)  |  ![](./figures/isrr2017_structured.gif)

J. Wang and B. Englot, "Autonomous Exploration with Expectation-Maximization," International Symposium on Robotics Research, Accepted, To Appear, December 2017.

# Usage

```
mkdir build && cd build
cmake ..
make

# add library to python path
export PYTHONPATH=build_dir

python ../script/isrr2017.py
```

> Note that figures are generated using OSX Clang. Results will be different with GCC.