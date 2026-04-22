# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for wp.config.enable_kernel_specialize."""

import os
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def axpy(y: wp.array(dtype=float), x: wp.array(dtype=float), alpha: float):
    i = wp.tid()
    y[i] = alpha * x[i] + y[i]


@wp.kernel
def axpy_2d(out: wp.array2d(dtype=float), a: wp.array2d(dtype=float), scale: float):
    i, j = wp.tid()
    out[i, j] = scale * a[i, j]


@wp.kernel
def bool_kernel(out: wp.array(dtype=float), flag: bool):
    i = wp.tid()
    if flag:
        out[i] = 1.0
    else:
        out[i] = -1.0


@wp.kernel
def scalar_only(n: int):
    pass


@wp.func
def inner_add(arr: wp.array(dtype=float), i: int) -> float:
    return arr[i]


@wp.func
def outer_scale(arr: wp.array(dtype=float), i: int, s: float) -> float:
    return inner_add(arr, i) * s


@wp.kernel
def nested_func_kernel(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = outer_scale(a, tid, 3.0)


@wp.func
def multi_return(a: float, b: float) -> tuple[float, float]:
    return a + b, a * b


@wp.kernel
def multi_return_kernel(x: wp.array(dtype=float), out_sum: wp.array(dtype=float), out_prod: wp.array(dtype=float)):
    i = wp.tid()
    s, p = multi_return(x[i], 2.0)
    out_sum[i] = s
    out_prod[i] = p


@wp.kernel
def strided_kernel(arr: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = arr[i] * 2.0


@wp.func
def scale_func(x: float, s: float) -> float:
    return x * s


@wp.kernel
def scalar_through_func(a: wp.array(dtype=float), out: wp.array(dtype=float), s: float):
    i = wp.tid()
    out[i] = scale_func(a[i], s)


def _find_spec_cu(kernel_key):
    """Find the generated .cu file for a specialized module."""
    for cache_root in [wp.config.kernel_cache_dir, os.path.dirname(wp.config.kernel_cache_dir)]:
        for dirpath, _dirnames, filenames in os.walk(cache_root):
            for f in filenames:
                if f.endswith(".cu") and "_spec_" in f and kernel_key in f:
                    return os.path.join(dirpath, f)
    return None


def _run_specialized(kernel, dim, inputs, device="cuda:0"):
    """Launch a kernel with specialization via graph capture and return the spec module name."""
    from warp._src.context import user_modules

    spec_before = {k for k in user_modules if "_spec_" in k}
    with wp.ScopedCapture(device=device) as cap:
        wp.launch(kernel, dim=dim, inputs=inputs, device=device)
    wp.capture_launch(cap.graph)
    wp.synchronize_device(device)
    spec_after = {k for k in user_modules if "_spec_" in k}
    new_specs = spec_after - spec_before
    return new_specs


class TestKernelSpecialize(unittest.TestCase):
    """Tests for kernel specialization."""

    def setUp(self):
        self.saved = wp.config.enable_kernel_specialize
        wp.config.enable_kernel_specialize = True

    def tearDown(self):
        wp.config.enable_kernel_specialize = self.saved

    # ---- Codegen verification ----

    def test_codegen_bakes_constants(self):
        """Verify the generated .cu has baked shape/stride/ndim, baked
        scalar constants, and scheme-B helpers for array access.
        """
        N = 256
        device = "cuda:0"
        x = wp.array(np.ones(N, dtype=np.float32), device=device)
        y = wp.array(np.ones(N, dtype=np.float32), device=device)

        new_specs = _run_specialized(axpy, dim=N, inputs=[y, x, 2.0], device=device)
        self.assertTrue(len(new_specs) > 0, "No specialized module was created")

        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path, f"Specialized .cu not found for {spec_name}")

        with open(cu_path) as f:
            source = f.read()

        # Baked dim and scalar.
        self.assertIn(f"dim.size = {N}", source)
        self.assertIn("var_alpha = 2", source)
        # Baked array metadata as prologue writes (kept for NVRTC alias
        # analysis — see codegen_kernel comment).
        self.assertIn(f".shape.dims[0] = {N}", source)
        self.assertIn(".strides[0] = 4", source)
        self.assertIn(".ndim = 1", source)
        # Scheme-B helpers: array accesses route through per-module baked
        # helpers that take `.data` directly.  In the template form
        # (Phase A), shape + stride are non-type template params, so the
        # helper itself has no hash suffix — one template per (builtin,
        # ndim) with many instantiations per module.
        self.assertIn("template<int S0, int St0, typename T>", source)
        self.assertRegex(source, rf"wp::wp_address_baked_1d<{N}, 4>\(")
        self.assertRegex(source, rf"wp::wp_array_store_baked_1d<{N}, 4>\(")

    def test_codegen_nested_func_variants(self):
        """Verify baked function variants are generated for nested wp.func calls."""
        N = 64
        device = "cuda:0"
        a = wp.array(np.ones(N, dtype=np.float32), device=device)
        out = wp.zeros(N, dtype=float, device=device)

        new_specs = _run_specialized(nested_func_kernel, dim=N, inputs=[a, out], device=device)
        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path)

        with open(cu_path) as f:
            source = f.read()

        # Both outer and inner should have baked variants
        self.assertIn("outer_scale_0_baked_", source)
        self.assertIn("inner_add_0_baked_", source)
        # The baked outer body should call the baked inner (not generic)
        import re

        self.assertRegex(source, re.compile(r"outer_scale_0_baked_\w+.*?inner_add_0_baked_\w+", re.DOTALL))

    # ---- Correctness (parametrized: specialized vs generic) ----

    def _assert_axpy(self, use_graph):
        N = 512
        device = "cuda:0"
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(N).astype(np.float32)
        y_np = rng.standard_normal(N).astype(np.float32)
        expected = 2.0 * x_np + y_np

        x = wp.array(x_np, device=device)
        y = wp.array(y_np.copy(), device=device)

        if use_graph:
            with wp.ScopedCapture(device=device) as cap:
                wp.launch(axpy, dim=N, inputs=[y, x, 2.0], device=device)
            wp.capture_launch(cap.graph)
        else:
            wp.launch(axpy, dim=N, inputs=[y, x, 2.0], device=device)
        wp.synchronize_device(device)

        np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)

    def test_correctness_graph(self):
        self._assert_axpy(use_graph=True)

    def test_correctness_no_graph(self):
        self._assert_axpy(use_graph=False)

    def test_correctness_disabled(self):
        wp.config.enable_kernel_specialize = False
        self._assert_axpy(use_graph=True)

    def test_2d_kernel(self):
        device = "cuda:0"
        a = wp.array(np.ones((32, 16), dtype=np.float32), device=device)
        out = wp.zeros((32, 16), dtype=float, device=device)

        with wp.ScopedCapture(device=device) as cap:
            wp.launch(axpy_2d, dim=(32, 16), inputs=[out, a, 3.0], device=device)
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)

        np.testing.assert_allclose(out.numpy(), 3.0, rtol=1e-5)

    def test_bool_arg(self):
        N = 64
        device = "cuda:0"
        out = wp.zeros(N, dtype=float, device=device)

        with wp.ScopedCapture(device=device) as cap:
            wp.launch(bool_kernel, dim=N, inputs=[out, True], device=device)
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)

        np.testing.assert_allclose(out.numpy(), 1.0)

    def test_zero_dim(self):
        device = "cuda:0"

        with wp.ScopedCapture(device=device) as cap:
            wp.launch(scalar_only, dim=0, inputs=[0], device=device)
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)

    def test_graph_replay(self):
        N = 64
        device = "cuda:0"
        x = wp.array(np.ones(N, dtype=np.float32), device=device)
        y = wp.array(np.ones(N, dtype=np.float32), device=device)

        with wp.ScopedCapture(device=device) as cap:
            wp.launch(axpy, dim=N, inputs=[y, x, 2.0], device=device)
        graph = cap.graph

        wp.capture_launch(graph)
        wp.synchronize_device(device)
        np.testing.assert_allclose(y.numpy()[0], 3.0, rtol=1e-5)

        wp.capture_launch(graph)
        wp.synchronize_device(device)
        np.testing.assert_allclose(y.numpy()[0], 5.0, rtol=1e-5)

    def test_module_reuse(self):
        from warp._src.context import user_modules

        N = 256
        device = "cuda:0"

        spec_before = sum(1 for k in user_modules if "_spec_" in k)

        x1 = wp.array(np.ones(N, dtype=np.float32), device=device)
        y1 = wp.array(np.ones(N, dtype=np.float32), device=device)
        with wp.ScopedCapture(device=device) as cap1:
            wp.launch(axpy, dim=N, inputs=[y1, x1, 2.0], device=device)

        spec_after_first = sum(1 for k in user_modules if "_spec_" in k)

        # Different pointers, same shapes — should reuse
        x2 = wp.array(np.ones(N, dtype=np.float32), device=device)
        y2 = wp.array(np.ones(N, dtype=np.float32), device=device)
        self.assertNotEqual(x1.ptr, x2.ptr)
        with wp.ScopedCapture(device=device) as cap2:
            wp.launch(axpy, dim=N, inputs=[y2, x2, 2.0], device=device)

        spec_after_second = sum(1 for k in user_modules if "_spec_" in k)
        self.assertEqual(spec_after_second, spec_after_first)

        # Different scalar — new module
        with wp.ScopedCapture(device=device) as cap3:
            wp.launch(axpy, dim=N, inputs=[y2, x2, 3.0], device=device)

        spec_after_third = sum(1 for k in user_modules if "_spec_" in k)
        self.assertEqual(spec_after_third, spec_after_first + 1)

    def test_nested_func(self):
        N = 256
        device = "cuda:0"
        a = wp.array(np.arange(N, dtype=np.float32), device=device)
        out = wp.zeros(N, dtype=float, device=device)

        with wp.ScopedCapture(device=device) as cap:
            wp.launch(nested_func_kernel, dim=N, inputs=[a, out], device=device)
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)

        np.testing.assert_allclose(out.numpy(), a.numpy() * 3.0, rtol=1e-5)

    def test_multi_return_func(self):
        N = 64
        device = "cuda:0"
        x = wp.array(np.arange(1, N + 1, dtype=np.float32), device=device)
        out_sum = wp.zeros(N, dtype=float, device=device)
        out_prod = wp.zeros(N, dtype=float, device=device)

        with wp.ScopedCapture(device=device) as cap:
            wp.launch(multi_return_kernel, dim=N, inputs=[x, out_sum, out_prod], device=device)
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)

        x_np = x.numpy()
        np.testing.assert_allclose(out_sum.numpy(), x_np + 2.0, rtol=1e-5)
        np.testing.assert_allclose(out_prod.numpy(), x_np * 2.0, rtol=1e-5)

    def test_non_contiguous_strides(self):
        device = "cuda:0"
        full = wp.array(np.arange(100, dtype=np.float32), device=device)
        strided = full[::2]
        out = wp.zeros(50, dtype=float, device=device)

        with wp.ScopedCapture(device=device) as cap:
            wp.launch(strided_kernel, dim=50, inputs=[strided, out], device=device)
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)

        expected = np.arange(0, 100, 2, dtype=np.float32) * 2.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_scalar_through_func(self):
        """Scalar arguments propagate as baked constants through wp.func."""
        N = 128
        device = "cuda:0"
        a = wp.array(np.arange(N, dtype=np.float32), device=device)
        out = wp.zeros(N, dtype=float, device=device)

        new_specs = _run_specialized(scalar_through_func, dim=N, inputs=[a, out, 5.0], device=device)
        self.assertTrue(len(new_specs) > 0)

        # Verify the baked function variant has the scalar constant
        cu_path = _find_spec_cu("scalar_through_func")
        self.assertIsNotNone(cu_path)
        with open(cu_path) as f:
            source = f.read()
        self.assertIn("scale_func_0_baked_", source)
        # The baked variant should assign the scalar value
        self.assertIn("var_s = 5", source)

        np.testing.assert_allclose(out.numpy(), a.numpy() * 5.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
