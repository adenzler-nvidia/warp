# NCU Profiling Instructions for grid-stride removal branch

## Branch

`adenzler/remove-grid-stride-v2` on `adenzler-nvidia/warp` (GitHub fork)

Base: `main` at `e9f0d31a`

## What changed

This branch replaces the grid-stride `for` loop in all CUDA kernel templates with a 3D grid launch + early return:

**Before (main):**
```cuda
for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
     _idx < dim.size;
     _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
{
    // body (return; converted to continue;)
}
```

**After (this branch):**
```cuda
size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y)
            * static_cast<size_t>(gridDim.x * blockDim.x)
            + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
if (_idx >= dim.size) return;
// body (return; stays as return;)
```

The 3D grid is sized so that `grid.x * grid.y * grid.z * block_dim >= dim`, giving every thread a unique index without looping. The 32-bit intermediate math keeps the index computation cheap (two 32-bit MADs + one widening multiply).

Files changed:
- `warp/_src/codegen.py` — kernel templates (forward + backward), removed `return→continue` conversion, removed `continue;` at end of reverse body
- `warp/native/warp.cu` — `wp_cuda_launch_kernel()` now computes a 3D grid instead of 1D
- `warp/tests/test_codegen.py` — removed `max_blocks=1` from one test that depended on grid-stride behavior

## Results so far

**SASS register counts (ptxas -v, sm_89):** 134/153 mujoco_warp kernels show reduced register counts. Highlights:
- `update_gradient_cholesky`: 144 → 48 regs
- `_tile_cholesky_factorize_solve`: 168 → 48 regs
- `_transmission`: 116 → 70 regs
- Most `forward`/`smooth`/`solver` kernels: -2 to -16 regs

**nsys kernel timing (humanoid 8192 worlds, 100 steps):** -6.2% total kernel time across top 35 kernels.

**Regressions (4 kernels):** These have FEWER instructions AND fewer registers, yet run SLOWER:

| Kernel | Main regs | New regs | Main SASS lines | New SASS lines | Time delta |
|--------|-----------|----------|-----------------|----------------|------------|
| `_geom_local_to_global_e28b714c` | 36 | 29 | 1077 | 869 | +43% |
| `_cinert_1261260c` | 34 | 35 | ~520 | ~430 | +30% |
| `_compute_body_inertial_frames_d3bdd81a` | 34 | 32 | ~1350 | ~1130 | +46% |
| `_compute_body_matrices_27760750` | 26 | 24 | ~1410 | ~1170 | +14% |

All four are at 100% theoretical occupancy in both variants (low register counts). All are 1D launches (~544 blocks, grid_y=1, grid_z=1) so the 3D grid machinery isn't even involved — the regression is purely from the template change (`if/return` vs `for` loop).

Hypothesis: ptxas generates different instruction scheduling for the `if/return` code structure vs the `for` loop structure, possibly related to memory prefetch or warp scheduling. Need ncu metrics to confirm.

## What needs to be profiled

Run ncu on the 4 regressing kernels listed above, comparing `main` vs this branch. The goal is to understand WHY fewer instructions + fewer registers = slower.

Suggested ncu sections:
- `SpeedOfLight` — overall compute/memory utilization
- `MemoryWorkloadAnalysis` — L1/L2 hit rates, bandwidth
- `SchedulerStats` — warp stall reasons
- `WarpStateStats` — warp occupancy, stall breakdown
- `InstructionStats` — instruction mix, IPC
- `ComputeWorkloadAnalysis` — pipe utilization

## Setup steps

1. **Clone and checkout:**
   ```bash
   git clone https://github.com/adenzler-nvidia/warp.git
   cd warp
   git checkout adenzler/remove-grid-stride-v2
   ```

2. **Build native library:**
   ```bash
   uv run build_lib.py --quick
   ```

3. **IMPORTANT: Clear kernel cache.** The template change is not captured in Warp's module hash, so stale cached `.cu`/`.cubin` files will have the old template:
   ```bash
   rm -rf ~/.cache/warp
   ```

4. **Verify the template is active** after running any Warp program:
   ```bash
   grep "blockIdx.z" $(find ~/.cache/warp -name "*.cu" | head -1)
   # Should show: size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + ...
   ```
   If it shows the old `for (size_t _idx = ...)` pattern, delete all `__pycache__` dirs and the kernel cache again.

5. **Install mujoco_warp** for benchmarking:
   ```bash
   cd /path/to/mujoco_warp
   pip install -e .
   pip install -e /path/to/warp  # or set PYTHONPATH=/path/to/warp
   ```

6. **Enable ncu access** (requires host-level change):
   ```bash
   # On the HOST (not in container):
   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
   sudo update-initramfs -u -k all  # Debian/Ubuntu
   sudo reboot
   # Verify: cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly  → should be 0
   ```

## Profiling commands

**Profile regressing kernels on this branch:**
```bash
rm -rf ~/.cache/warp
ncu --set detailed --graph-profiling node \
    --kernel-name "regex:geom_local_to_global_e28b714c|cinert_1261260c|compute_body_inertial_frames_d3bdd81a|compute_body_matrices_27760750" \
    --launch-count 4 \
    -o ncu_3d_regressing \
    mjwarp-testspeed benchmarks/humanoid/humanoid.xml \
    --nworld=8192 --nconmax=24 --njmax=64 --nstep=5
```

**Profile same kernels on main (baseline):**
```bash
git checkout main
uv run build_lib.py --quick
rm -rf ~/.cache/warp
ncu --set detailed --graph-profiling node \
    --kernel-name "regex:geom_local_to_global_e28b714c|cinert_1261260c|compute_body_inertial_frames_d3bdd81a|compute_body_matrices_27760750" \
    --launch-count 4 \
    -o ncu_main_regressing \
    mjwarp-testspeed benchmarks/humanoid/humanoid.xml \
    --nworld=8192 --nconmax=24 --njmax=64 --nstep=5
```

**Compare reports:**
```bash
ncu --import ncu_main_regressing.ncu-rep --import ncu_3d_regressing.ncu-rep --page details
```

Or use the Nsight Compute UI to open both reports side by side.

## Key questions for the ncu analysis

1. **Is the regression memory-bound or compute-bound?** Check SpeedOfLight compute vs memory utilization.
2. **Are there more stalls?** Check SchedulerStats for stall reasons (long scoreboard, memory dependency, etc.).
3. **Did IPC change?** Check InstructionStats for instructions per cycle.
4. **L1/L2 cache behavior?** The `if/return` structure might change memory access patterns due to different warp convergence at the exit point.
5. **Warp divergence?** The early `return` causes some threads to exit while others continue — check if this differs from the `for` loop where all threads stay in the loop body.

## Known issues

- **NVRTC large-module bug:** In modules with 53+ kernels and complex inlined templates (like `wp::view()`), NVRTC 12.6 miscompiles the `if/return` pattern, producing wrong results. This was confirmed on `test_array3d_slicing` in `test_array.py` (17K lines, 53 kernels). mujoco_warp modules are smaller (~15K lines, ~45 kernels) and work correctly. The bug does NOT affect mujoco_warp.

- **`max_blocks` parameter:** The branch ignores `max_blocks` since without the grid-stride loop, capping blocks drops work. This is a behavioral change for the public API.

- **Module hash doesn't capture template changes:** Warp's kernel cache keys on the function AST hash, not the template string. After switching branches, you MUST clear `~/.cache/warp` to get fresh compilation. A proper fix would hash the template into the module hash.
