# VQE Controller Workload

`vqe_controller_workload.cpp` is a C++17 program that generates a **synthetic VQE-driven Host–Controller workload**. It models the number of algorithm/measurement events and, optionally, ordinary memory accesses representing parameter, command, and readout traffic.

**This is not a QPU simulator or an instruction-accurate QICK controller model.** No CUDA-Q, GPU, or quantum hardware is required.

## Build and run

```bash
g++ -std=c++17 -O2 vqe_controller_workload.cpp -o vqe_controller_workload

# Show calculated event counts and assumed transfer volumes only
./vqe_controller_workload --execute=0

# Also execute synthetic buffer copies and per-shot local memory accesses
./vqe_controller_workload --execute=1

# List available options
./vqe_controller_workload --help
```

## Workload structure

For each objective callback, the program models:

1. A **simplified** optimizer parameter update (not actual L-BFGS).
2. One base energy evaluation, `observe(x)`.
3. Two additional energy evaluations per parameter for central-difference gradients, `observe(x ± h e_i)`.
4. For every `observe()`, `G` measurement groups, each with `S` shots.

The resulting event counts are:

```text
observes           = objectives × (1 + 2 × params)
group submissions  = observes × groups
shot executions    = group submissions × shots
```

These shot executions are **modeled events**, not physical quantum operations performed by the program.

## Configuration

| Option | Default | Meaning |
|---|---:|---|
| `--objectives=N` | 26 | Objective callbacks (inferred for the example VQE run) |
| `--params=N` | 36 | Variational parameters |
| `--groups=N` | 5 | Measurement groups per `observe()` (assumption) |
| `--shots=N` | 100 | Shots per group (assumption) |
| `--param-policy=full\|delta` | `full` | Full vector for every observe, or full vector for base observe and delta update for shifted observes |
| `--readout-policy=shot\|aggregate` | `aggregate` | Transfer per shot, or one transfer per completed group |
| `--full-param-bytes=N` | 288 | Full parameter payload in bytes |
| `--delta-param-bytes=N` | 16 | Assumed delta payload in bytes |
| `--command-bytes=N` | 32 | Assumed command payload per group |
| `--readout-shot-bytes=N` | 8 | Assumed readout payload per shot |
| `--readout-agg-bytes=N` | 64 | Assumed aggregated payload per group |
| `--execute=0\|1` | `1` | Count only, or also execute synthetic memory accesses |

## Example experiments

```bash
# Default event model: 26 × (1 + 2×36) = 1,898 observe events
./vqe_controller_workload --execute=0

# Compare full-vector vs delta-only updates
./vqe_controller_workload --param-policy=delta --execute=0

# Compare per-shot and group-level aggregated readout
./vqe_controller_workload --readout-policy=shot --execute=0

# Larger shot count; actual memory accesses enabled
./vqe_controller_workload --groups=5 --shots=1000 --execute=1
```

With defaults, the model reports **1,898 observe events**, **9,490 group submissions**, and **949,000 shot events**. With group-aggregated readout, it models **9,490 readout transfers**; with shot-level readout, **949,000 transfers**.

## Synthetic memory buffers

```text
Host → Controller: host_tx       → controller_rx  (parameters / commands)
Controller local:   controller_local              (per-shot buffer access)
Controller → Host: controller_tx → host_rx         (readout results)
```

The program uses `std::memcpy` for transfers and touches a 4 KiB `controller_local` buffer for each shot when `--execute=1`. The buffers are all **ordinary C++ process memory**. They do not correspond to actual controller SRAM, device MMIO, DMA engines, or QPU memory. The model also does **not** derive the outgoing `controller_tx` data from `controller_local`; it simulates their traffic separately.

## Interpretation and limitations

- The 26 objective callbacks are **inferred** from prior CUDA-Q source analysis and Nsight kernel patterns, not independently counted.
- `G`, `S`, transfer policies, and most payload sizes are **configurable modeling assumptions**, not measured QICK specifications.
- The program does not implement Pauli grouping, quantum gates, waveforms, measurements, or physical timing.
- In gem5 syscall-emulation (SE) mode, memory operations appear as ordinary CPU-process accesses, **not** real Host–Controller device transactions. Device-level latency, DMA, or SRAM modeling would require additional gem5 hardware abstractions.

**Intended use:** characterize workload scaling and compare hypothetical communication policies before attempting a more detailed architecture model.
