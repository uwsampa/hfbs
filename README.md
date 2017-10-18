# HFBS: hardware-friendly bilateral solver

The hardware-friendly bilateral solver (HFBS) is a parallel algorithm for bilateral solving.
This repo has Python software reference and Verilog hardware implementations for experimentation.
The Verilog implementation comes with scaffolding for deployment on a Xilinx Zynq ZC702 dev board.
The software code is an unoptimized version of the implementation described in the paper, so runtimes are worse than those reported in the paper.

You can read more about HFBS in the [HPG 2017 paper](https://homes.cs.washington.edu/~amrita/papers/hpg17.pdf) or the [summary blog post](http://amritamaz.net/blog/hpg17-hfbs).

## Getting Started

HFBS uses Python 2.7 with numpy, matplotlib, and scipy.

You can test the software reference by running

```
python hfbs.py
```

which will run HFBS with the input data from the `data/depth_superres` folder.

## Building and running the FPGA design

(**unstable, under construction**)

To build the hardware project, open it in Vivado and compile.

Program the FPGA and set up the driver.

Once the FPGA is programmed and ready, you can invoke the bilateral solver from the software reference.

```
python hfbs_fpga.py
```

which should display the input and processed output.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
