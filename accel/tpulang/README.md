# tpulang & pytpu

Pytpu python-based language used for writing kernels on tpu hardware. It is compiled into tpulang, which is then assembled
into tpu instruction code.

## pytpu overview

### Tensor

Tensor class represents a tensor (either int8 or ternary) in tpu memory. Fields: 
- dtype (int8, int2): tensor dtype.
- shape (Tuple[int]): the shape of the tensor.
- scale (float): the quantization scale. This value is NOT stored in TPU RAM. It is only used during compile time
to determine requantization in matmul.

### Scalar

The scalar class represents a sequence 32 bits stored in memory. It can either be an int32 or a tuple {M0, n} where 
M0 is a 12-bit int and n is a 4 bit int, representing a requantization factor.

### Primitives

Primitives are tensor operations that can be directly mapped to a single TPU instruction. Examples include matmul,
vector addition & dot products, and ReLU.
All other operations should be reduced into a sequence of primitives. For example, softmax can be decomposed into 
a max reduction, scalar subtraction, exp, sum reduction, and scalar division.

#### Matmuls

The compiler automatically tiles (ternary * int8) matmuls and expands (int8 * int8) matmuls into VPU dot product operations. 

### Memory Management

TPU architecture contains 2 levels of memory: RAM and scratchpad memory. Movement to/from scratchpad (Write/Read Memory instructions)
handled by the compiler when calling a primitive. We will consider implementing automatic fusion to prevent redundant
memory access later.


## tpulang

### Overview

tpulang serves as an assembly language for the TPU hardware and has a direct mapping to the instruction set.







