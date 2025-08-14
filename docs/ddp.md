# Blackwell Support




## Problem Description

Getting tch-gpu working on Blackwell, have 5090.   Need torch 2.8 for support for sm_120.

Update tch-rs to 0.21 (unreleased version current)

```bash
cargo run --example mnist --release --features tch-gpu
```

## Main Error

When running DDP training with GPU backends, users may encounter the following error:

```
called `Result::unwrap()` on an `Err` value: Torch("Input type (CPUFloatType) and weight type (CUDAFloatType) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
Exception raised from check_input_same_type_as_parameters at /pytorch/aten/src/ATen/native/Convolution.cpp:789
```

This error occurs specifically at the first convolution operation during training, indicating that input tensors are on CPU while model weights are on CUDA devices.

### Root Cause Analysis

The issue lies in the device coordination mechanism within Burn's DDP implementation. Here's the detailed breakdown:

#### 1. DDP Setup Process

In `crates/burn-train/src/learner/strategies/ddp/method.rs:41`, the training dataloader is split across devices:

```rust
let train = split_dataloader(dataloader_train, &self.devices);
```

This calls `split_dataloader()` in `crates/burn-core/src/data/dataloader/split.rs:25`:

```rust
let dataloader = dataloader.slice(start, end).to_device(device);
```

#### 2. Device Assignment Bug

The issue occurs in the interaction between `slice()` and `to_device()` methods:

**In `crates/burn-core/src/data/dataloader/batch.rs:119-132`:**

```rust
fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, O>> {
    let dataloader = Self::new(
        self.strategy.clone_dyn(),
        Arc::new(PartialDataset::new(self.dataset.clone(), start, end)),
        self.batcher.clone(),
        self.device.clone(), // ← BUG: Uses original device, not target device
        rng,
    );
    Arc::new(dataloader)
}
```

When `dataloader.slice(start, end).to_device(device)` is called:
1. `slice()` creates a new dataloader with the **original device** (`self.device.clone()`)
2. `to_device()` is supposed to fix this by setting the correct device
3. However, there's a coordination issue where the device assignment doesn't properly propagate to the batching process

#### 3. Runtime Execution

During training in `crates/burn-train/src/learner/strategies/ddp/epoch.rs:114`:

```rust
let item = model.step(item);
```

The `item` (batch data) comes from the dataloader iterator, but due to the device coordination bug, tensors may still be created on CPU instead of the intended GPU device. Meanwhile, the model has been correctly moved to the GPU via `model.fork(device)`.

#### 4. PyTorch Error

The actual error originates from PyTorch's strict device consistency checks in the underlying `atg_conv2d` function called through tch-rs (`src/wrappers/tensor_fallible_generated.rs:13520`). PyTorch requires input and weight tensors to be on the same device for convolution operations.

## Workaround Solution

### For Single-GPU Training

Replace DDP with SingleDevice learning strategy:

**Before (problematic):**
```rust
.learning_strategy(burn::train::ddp(vec![device], collective))
```

**After (working):**
```rust
.learning_strategy(burn::train::LearningStrategy::SingleDevice(device))
```

This bypasses the DDP device splitting entirely, ensuring all data and model components remain on the same device.


### Long-term
1. Fix the device coordination in `split_dataloader` and related methods
2. Add comprehensive device consistency tests for DDP scenarios
3. Implement better error handling and device validation in DDP setup

## Testing

### Verification Steps

1. **Reproduce Issue**: Run MNIST example with tch-gpu and DDP
2. **Verify Fix**: Run same example with SingleDevice strategy
3. **Performance Check**: Ensure no performance regression with SingleDevice

### Test Commands

```bash
# This will fail with device mismatch
cargo run --example mnist --features tch-gpu  # with DDP

# This will work correctly  
cargo run --example mnist --features tch-gpu  # with SingleDevice
```

## References

- **Issue Location**: `crates/burn-core/src/data/dataloader/batch.rs:128`
- **DDP Implementation**: `crates/burn-train/src/learner/strategies/ddp/`
- **Error Source**: PyTorch's `Convolution.cpp:789` device consistency check
- **tch-rs Integration**: `src/wrappers/tensor_fallible_generated.rs:13520`

This issue demonstrates the complexity of device coordination in distributed training scenarios and highlights the importance of thorough device consistency testing across different backend implementations.