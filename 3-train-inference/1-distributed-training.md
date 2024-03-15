# Distributed Training

We need to be aware of what kind of distributed learning we can use, and there are `DDP`, `RPC`, and `Collective communication` from the [pytorch documentation](https://pytorch.org/tutorials/beginner/dist_overview.html) (read the documentation for the details).

- **[Distributed Data-Parallel Training](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP)**
- **[RPC-Based Distributed Training](https://pytorch.org/docs/stable/rpc.html) (RPC)**
- **[Collective Communication](https://pytorch.org/docs/stable/distributed.html) (c10d)**

# **DP/ Data Parallel**

- **Concept**: a single model is replicated across multiple devices. Each device processes a `different subset of the data` but uses an `identical copy of the model`. After processing, the gradients from all devices are aggregated (usually by summing) and then used to update the model parameters synchronously.
- **Usage Scenario**: `the model and the data` can fit entirely into the memory of each device.
- **Limitations**: The main limitation is the scaling inefficiency due to the replication of the model across all devices. This means each device must have enough memory to store the entire model and a batch of data, which becomes impractical for very large models.

# **DDP/ Distributed Data Parallel**

- **Concept**: Same as DP, a single model is replicated across multiple devices. Each device processes a `different subset of the data` but uses an `identical copy of the model`.. However, DDP is optimized for multi-node setups and reduces the communication overhead by intelligently managing `gradient synchronization`. Each process (device) performs gradient computation independently.
- **Usage Scenario**: DDP is highly efficient in terms of parallelization and communication, especially when using high-speed network interconnects.
- **Advantages Over DP**: DDP reduces the communication bottleneck by using more efficient collective communication primitives and overlapping communication with computation.

# FSDP / **Fully Sharded Data Parallel**

- **Concept**: Instead of replicating the entire model on every device, FSDP shards (splits) the model parameters, gradients, and optimizer states across all available devices. This means `each device only stores a fraction of the model`.
- **Usage Scenario**: Ideal for very large models that cannot fit into the memory of a single device.
- **Advantages Over DDP**: FSDP significantly reduces the per-device memory requirements, enabling the training of much larger models. It also incorporates techniques like `mixed precision training` and `gradient checkpointing` to further optimize memory usage and computational efficiency.

### **Summary of Differences**

- **Scalability**: DP is less scalable than DDP and FSDP due to its inefficient use of memory and compute resources.
- **Memory Efficiency**: FSDP > DDP > DP, with FSDP being the most memory-efficient, allowing for training larger models.
- **Complexity and Overhead**: FSDP is the most complex in terms of implementation and has additional overhead due to model sharding and reconstruction during the forward and backward passes. DDP is less complex than FSDP but more so than DP, which is the simplest.

## **[Typical Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp_examples.html#id2)**

Typical Mixed Precision Training in PyTorch refers to a training method that uses both 16-bit (half-precision) and 32-bit (single-precision) floating-point arithmetic to accelerate the training of neural networks while maintaining the model's accuracy. 

# RPC

(RPC) supports general training structures that cannot fit into data-parallel training such as distributed pipeline parallelism, parameter server paradigm, and combinations of DDP with other training paradigms.

# Reference
- [Docker Example](https://github.com/pytorch/elastic/blob/master/examples/imagenet/main.py)
- [k8s Example](https://github.com/pytorch/elastic/tree/master/kubernetes)