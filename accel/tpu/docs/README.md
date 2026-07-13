# TPU Design Doc

## Component docs

Detailed per-component design notes (this file is the overview):

| Doc                              | Component                                              |
| -------------------------------- | ----------------------------------------------------- |
| [scratchpad.md](scratchpad.md)   | On-chip BRAM working memory, banking, sizing          |
| [mxu.md](mxu.md)                 | Weight-stationary ternary systolic matrix unit        |
| [vpu.md](vpu.md)                 | SIMD vector unit: activations, reductions, attention  |
| [comms.md](comms.md)             | 2D inter-TPU link for scale-out                       |
| [scalar_unit.md](scalar_unit.md) | Control processor + ISA                               |

## 1. Overview

This is a design for a TPU that goes on low-end FPGA boards for inferencing transformers.
To minimize area usage, we target highly quantized neural networks: those with weights in ${1, 0, -1}$ 
and 8-bit activations. We target running a ternary version of the digit-predicting adder LLM. We also design for scale-out,
allowing multiple devices to communicate during inference.

We use the same general architecture as google TPU:

### 1. Scratchpad

We will use DMA to access RAM and load into a specific address in our scratchpad memory, which uses FPGA BRAM. 

### 2. MXU

The MXU is a weight-stationary systolic array. It will load the ternary weights from scratchpad memory into registers in each PE.
Then, it loads activations from the scratchpad into the activation buffer and feeds them into the systolic array in a staggered order. 
The scratchpad should consist of enough BRAM blocks to fully load 1 column of activations in 1 clock.
The MXU output immediately gets written to another address in scratchpad memory.

### 3. VPU

The VPU performs all other important pointwise vector operations using SIMD, such as activations (relu, gelu, etc), vector add, dot product, and reductions. 
It contains multiple ALUs that act on data from a single scratchpad memory access.

### Requant unit

All tensors outputted by the MXU or VPU need to be requantized according to a channelwise scaling M, determined during model quantization. 

### 4. Communication interface

Furthermore, we implement an inter-TPU interface that allows a TPU to receive requests to write to the RAM of its neighbors. For now, we implement a 2d connection scheme (4 neighbors).


### 4. Scalar Unit

Handles all the control, dispatches instructions to the VPU and MXU. Reads from instruction memory.
Also handles scalar operations such as adding, multiplying numbers in memory.

Instructions:

Math (all addresses are within scratchpad memory):
Matmul(activation addr, weight addr, out addr): performs activation @ weight -> out, fixed size
VectorDot(vector addr 1, vector addr 2, out addr): 
VectorMultiply(vector addr, scalar addr, out addr)
VectorAdd(vector addr 1, vector addr 2, out addr)
ReLU(vector pad addr, out addr)
GeLU(vector addr, out addr)

Comms/Memory (all addresses are in ram unless stated)
WriteNeighbor(neighbor, my addr, neighbor addr): uses inter-TPU interface to write to neighbor's RAM
WriteMemory(scratchpad (read) addr, write addr)
ReadMemory(read addr, scratchpad (write) addr)







