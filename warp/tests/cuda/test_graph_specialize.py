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


@wp.kernel
def reads_shape(a: wp.array(dtype=float), out: wp.array(dtype=int)):
    tid = wp.tid()
    if tid == 0:
        out[0] = a.shape[0]


# Exercises the runtime-K path: `a.shape[k]` with `k` from a non-
# unrollable source (the volatile load forces ptxas to keep `k` runtime).
# The kernel returns 1 if the runtime-K access produces the expected dim.
@wp.kernel
def reads_shape_runtime(a: wp.array2d(dtype=float), k_in: wp.array(dtype=int), out: wp.array(dtype=int)):
    tid = wp.tid()
    if tid == 0:
        k = k_in[0]
        out[0] = a.shape[k]


# Exercises arr.ndim on a baked kernel-arg array.
@wp.kernel
def reads_ndim(a: wp.array2d(dtype=float), out: wp.array(dtype=int)):
    tid = wp.tid()
    if tid == 0:
        out[0] = a.ndim


# Exercises arr.strides[K] (literal K) and arr.strides[k] (runtime K).
@wp.kernel
def reads_strides(
    a: wp.array2d(dtype=float),
    k_in: wp.array(dtype=int),
    out: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid == 0:
        k = k_in[0]
        out[0] = a.strides[0]
        out[1] = a.strides[k]


@wp.kernel
def views_2d(arr: wp.array2d(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    row = arr[tid]
    out[tid] = row[0]


@wp.kernel
def where_on_array(maybe: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.where(maybe, 1.0, 2.0)


@wp.kernel
def nested_view(arr: wp.array3d(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    row = arr[tid]
    inner = row[0]
    out[tid] = inner[0]


@wp.kernel
def view_then_shape(arr: wp.array2d(dtype=float), out: wp.array(dtype=int)):
    tid = wp.tid()
    if tid == 0:
        slice = arr[0]
        out[0] = slice.shape[0]


# Exercises the Phase D body-sharing path: a wp.func that views one of
# its array args and reads the view-result's shape, called twice from
# the same kernel with two differently-shaped arrays.  Phase D shares
# one template body across both bakings via Config:: traits — without
# provenance tracking on view-result Vars, the body would bake in the
# first caller's literal and the second call would silently get the
# wrong answer.
@wp.func
def view_inner_shape(a: wp.array2d(dtype=float)) -> int:
    s = a[0]
    return s.shape[0]


@wp.kernel
def view_inner_shape_caller(
    a: wp.array2d(dtype=float),
    b: wp.array2d(dtype=float),
    out: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid == 0:
        out[0] = view_inner_shape(a)
        out[1] = view_inner_shape(b)


# Per-kernel opt-out — even when ``enable_kernel_specialize`` is True
# globally, this kernel must NEVER take the spec path.
@wp.kernel(specialize=False)
def axpy_no_spec(y: wp.array(dtype=float), x: wp.array(dtype=float), alpha: float):
    i = wp.tid()
    y[i] = alpha * x[i] + y[i]


@wp.kernel
def scale_for_ad(x: wp.array(dtype=float), out: wp.array(dtype=float), alpha: float):
    i = wp.tid()
    out[i] = alpha * x[i]


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

        # Templated kernel: every baked value is a non-type template
        # parameter on the kernel template.
        self.assertRegex(source, r"template <int dim_shape_0,.*int y_shape_0,.*wp::float32 alpha>")
        # Body uses bare template-arg refs (no Python literal substitution).
        self.assertIn("dim.size = dim_size;", source)
        self.assertIn("var_alpha = alpha;", source)
        # Baked array ABI: kernel receives `T* __restrict__` directly.
        self.assertRegex(source, r"wp::float32\* __restrict__ var_y_data")
        self.assertRegex(source, r"wp::float32\* __restrict__ var_x_data")
        # Array access goes through ``wp::address(var_x, ...)`` and
        # ``wp::array_store(var_y, ...)``; C++ overload resolution picks
        # the templated ``baked_array_t<...>`` overload, which delegates
        # to the shape/stride-templated address helpers in ``array.h``.
        self.assertRegex(source, r"wp::address\(var_x, ")
        self.assertRegex(source, r"wp::array_store\(var_y, ")
        # The materialized baked_array_t locals carry the NTTPs in
        # their type, so the address helpers see them via overload
        # resolution rather than Python emitting them explicitly.
        self.assertRegex(source, r"wp::baked_array_t<wp::float32, x_ndim, x_shape_0,")
        self.assertRegex(source, r"wp::baked_array_t<wp::float32, y_ndim, y_shape_0,")

    def test_codegen_baked_shape_local(self):
        """Verify `arr.shape[K]` on a baked array uses a bare
        template-arg reference (the kernel is templated; NVRTC
        substitutes per-instantiation), with no shape_t local
        materialised.  End-to-end the kernel reads back the correct
        value.
        """
        N = 123
        device = "cuda:0"
        a = wp.zeros(N, dtype=float, device=device)
        out = wp.zeros(1, dtype=int, device=device)

        new_specs = _run_specialized(reads_shape, dim=N, inputs=[a, out], device=device)
        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path, f"Specialized .cu not found for {spec_name}")

        with open(cu_path) as f:
            source = f.read()

        # No materialised shape local.
        self.assertNotIn("__wp_baked_var_a_shape", source)
        # Body references the template arg, not a literal.
        self.assertRegex(source, r"var_\d+ = a_shape_0;")

        # End-to-end: the kernel reads back N correctly.
        wp.synchronize_device(device)
        self.assertEqual(int(out.numpy()[0]), N)

    def test_codegen_baked_shape_runtime_k(self):
        """Verify `arr.shape[k]` with runtime ``k`` dispatches through
        the templated ``wp::baked_shape_extract<S0, S1, S2, S3>(k)``
        free function, not through a materialised ``shape_t`` local.
        Static shape values arrive as template args; the runtime
        ternary collapses over compile-time constants.
        """
        N, M = 17, 32
        device = "cuda:0"
        a = wp.zeros((N, M), dtype=float, device=device)
        k_in = wp.array([1], dtype=int, device=device)  # k=1 → expect M
        out = wp.zeros(1, dtype=int, device=device)

        new_specs = _run_specialized(
            reads_shape_runtime, dim=N, inputs=[a, k_in, out], device=device
        )
        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path)

        with open(cu_path) as f:
            source = f.read()

        # Templated kernel: runtime-K shape access uses
        # baked_shape_extract with template-arg refs for the in-use
        # dims (slots beyond ndim are zero-padded).
        self.assertRegex(source, r"wp::baked_shape_extract<a_shape_0, a_shape_1, 0, 0>\(")
        # No materialised shape_t local for this kernel.
        self.assertNotIn("__wp_baked_var_a_shape", source)

        # End-to-end: the kernel reads back M (since k=1).
        wp.synchronize_device(device)
        self.assertEqual(int(out.numpy()[0]), M)

    def test_codegen_baked_ndim(self):
        """Verify `arr.ndim` on a baked kernel-arg array constant-folds
        to the literal at codegen time without going through any struct
        field access.  Without the spec hook, codegen would emit
        `var_a.ndim` which fails compile under T*-ABI (no `var_a`).

        ndim is type-level (fixed by the Warp annotation), so we emit
        the literal directly — the bare-NTTP-ref form used for
        shape/stride NTTPs would resolve to the same value at every
        instantiation anyway.
        """
        N, M = 17, 32
        device = "cuda:0"
        a = wp.zeros((N, M), dtype=float, device=device)
        out = wp.zeros(1, dtype=int, device=device)

        new_specs = _run_specialized(reads_ndim, dim=N, inputs=[a, out], device=device)
        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path)

        with open(cu_path) as f:
            source = f.read()

        # arr.ndim emits as a const literal (2 for ``wp.array2d``).
        self.assertRegex(source, r"const wp::int32 var_\d+ = 2;")
        # No struct-field access on var_a.
        self.assertNotIn("var_a.ndim", source)

        wp.synchronize_device(device)
        self.assertEqual(int(out.numpy()[0]), 2)

    def test_codegen_baked_strides(self):
        """Verify ``arr.strides[K]`` (literal K) constant-folds to the
        literal stride, and ``arr.strides[k]`` (runtime k) dispatches
        through the templated ``wp::baked_shape_extract<...>`` free
        function with the stride values as template args.
        """
        N, M = 17, 32
        device = "cuda:0"
        a = wp.zeros((N, M), dtype=float, device=device)
        k_in = wp.array([1], dtype=int, device=device)
        out = wp.zeros(2, dtype=int, device=device)

        new_specs = _run_specialized(
            reads_strides, dim=N, inputs=[a, k_in, out], device=device
        )
        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path)

        with open(cu_path) as f:
            source = f.read()

        # Row-major (N, M) float32 array → strides[0] = M * 4, strides[1] = 4.
        st0 = M * 4
        st1 = 4
        # Templated kernel: strides[K] reads the template arg ref.
        self.assertRegex(source, r"var_\d+ = a_stride_0;")
        # Runtime-K path: baked_shape_extract with stride template-arg refs.
        self.assertRegex(
            source, r"wp::baked_shape_extract<a_stride_0, a_stride_1, 0, 0>\("
        )

        wp.synchronize_device(device)
        self.assertEqual(int(out.numpy()[0]), st0)
        self.assertEqual(int(out.numpy()[1]), st1)

    def test_codegen_baked_array_for_view(self):
        """`arr[i]` on a 2D baked array takes the `view(arr, int)` path.
        Verify the emit uses `wp::baked_array_t<...>` with template-
        encoded shape/strides/ndim, the int-indexed view fires (no
        slice_t wrapping), and the result is correct.
        """
        N = 32
        M = 4
        device = "cuda:0"
        arr = wp.array(np.arange(N * M, dtype=np.float32).reshape(N, M), device=device)
        out = wp.zeros(N, dtype=float, device=device)

        new_specs = _run_specialized(views_2d, dim=N, inputs=[arr, out], device=device)
        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path)

        with open(cu_path) as f:
            source = f.read()

        # Templated kernel: kernel-arg materialized as baked_array_t
        # with template-arg refs for the in-use dims (extra slots
        # zero-padded since `bake_array_struct_decl` only fills 0..ndim).
        self.assertRegex(
            source,
            r"wp::baked_array_t<wp::float32, arr_ndim, arr_shape_0, arr_shape_1, 0, 0, arr_stride_0, arr_stride_1, 0, 0>",
        )
        # Codegen passes the bare int index through to view (rather than
        # wrapping in slice_t(i, i, 0)).
        self.assertRegex(source, r"wp::view\(var_arr, var_\d+\);")
        self.assertNotIn("wp::slice_t(var_", source)

        # End-to-end correctness.
        wp.synchronize_device(device)
        np.testing.assert_allclose(out.numpy(), np.arange(N, dtype=np.float32) * M)

    def test_codegen_baked_array_for_where(self):
        """`wp.where(arr, a, b)` forces baked_array_t reconstruction —
        the where overload is templated on baked_array_t so shape /
        stride / ndim flow through as compile-time template args.
        """
        N = 16
        device = "cuda:0"
        maybe = wp.array(np.arange(N, dtype=np.float32), device=device)
        out = wp.zeros(N, dtype=float, device=device)

        new_specs = _run_specialized(where_on_array, dim=N, inputs=[maybe, out], device=device)
        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path)

        with open(cu_path) as f:
            source = f.read()

        # Templated kernel: kernel-arg materialized as baked_array_t
        # with template-arg refs.
        self.assertRegex(
            source,
            r"wp::baked_array_t<wp::float32, maybe_ndim, maybe_shape_0,",
        )
        self.assertIn("wp::where(var_maybe,", source)

        wp.synchronize_device(device)
        np.testing.assert_allclose(out.numpy(), np.ones(N, dtype=np.float32))

    def test_codegen_nested_view_propagates_scheme_b(self):
        """`arr[i][j][k]` on a baked 3D kernel arg goes through two
        `wp::view(baked_array_t, int)` calls returning shape-shifted
        baked_array_t sub-arrays, then a final element access.  Verify
        the codegen tracks the sub-array shape/stride at each hop and
        emits scheme-B (`wp_address_baked_1d<S, St>(sub.data, ...)`)
        for the terminal access — i.e. static info flows end-to-end
        across nested views, not just from the kernel arg.
        """
        N, M, K = 8, 4, 5
        device = "cuda:0"
        arr = wp.array(np.arange(N * M * K, dtype=np.float32).reshape(N, M, K), device=device)
        out = wp.zeros(N, dtype=float, device=device)

        new_specs = _run_specialized(nested_view, dim=N, inputs=[arr, out], device=device)
        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path)

        with open(cu_path) as f:
            source = f.read()

        # Two int-indexed views chained off the kernel arg.  Kernel-arg
        # materialized via templated `baked_array_t<...>` with template-arg refs.
        self.assertRegex(
            source,
            r"wp::baked_array_t<wp::float32, arr_ndim, arr_shape_0,.*> var_arr",
        )
        # View-result locals declared via decltype(wp::view(...)) — C++
        # template deduction picks the post-view baked_array_t<...> type.
        self.assertRegex(source, r"decltype\(wp::view\(var_arr, 0\)\) var_\d+;")
        self.assertRegex(source, r"decltype\(wp::view\(var_\d+, 0\)\) var_\d+;")
        self.assertRegex(source, r"wp::view\(var_arr, var_\d+\);")
        self.assertRegex(source, r"wp::view\(var_\d+, var_\d+\);")
        # Terminal element access on the (sub-array of sub-array) uses
        # standard wp::address; overload resolution routes to the baked
        # variant in array.h via the local's templated type.  No
        # codegen-emitted scheme-B helper for view-results.
        self.assertRegex(source, r"wp::address\(var_\d+, var_\d+\);")

        wp.synchronize_device(device)
        np.testing.assert_allclose(out.numpy(), np.arange(N, dtype=np.float32) * M * K)

    def test_codegen_view_result_shape_baked(self):
        """``slice = arr[i]; slice.shape[0]`` on a baked source: the
        view result carries its post-view ``baked_value`` (shape ``(M,)``
        after viewing the 0th row of an ``(N, M)`` array), and
        ``slice.shape[0]`` constant-folds to ``M`` at codegen time
        without any shape_t materialisation.
        """
        N = 8
        M = 5
        device = "cuda:0"
        arr = wp.zeros((N, M), dtype=float, device=device)
        out = wp.zeros(1, dtype=int, device=device)

        new_specs = _run_specialized(view_then_shape, dim=N, inputs=[arr, out], device=device)
        spec_name = next(iter(new_specs))
        cu_path = _find_spec_cu(spec_name.replace(".", "_"))
        self.assertIsNotNone(cu_path)

        with open(cu_path) as f:
            source = f.read()

        # Templated kernel: view-result `.shape[K]` reads the typed
        # local's static accessor (`baked_shape[K]`); NVRTC folds to
        # the right per-instantiation value.  No materialised shape_t.
        self.assertNotIn("__wp_baked_var_", source.split("// shared memory")[-1] if "shared memory" in source else "")
        self.assertRegex(source, r"var_\d+ = var_\d+\.baked_shape\[0\];")

        wp.synchronize_device(device)
        self.assertEqual(int(out.numpy()[0]), M)

    def test_view_result_shape_in_templated_func_two_callers(self):
        """Phase D body-sharing correctness: a wp.func that views one
        of its array args and reads the view-result's shape, called
        twice from the same kernel with arrays of different shapes,
        must produce the right answer for each call — not bake the
        first caller's literal into the shared body.

        Phase D registers ONE C++ template per (function, label-set)
        and instantiates it with different ``Config`` traits per call.
        Without provenance tracking on view-result Vars,
        ``s.shape[0]`` inside ``view_inner_shape`` would emit
        ``const int = M1`` from whichever call was built first, and
        the second call would silently get the wrong answer.
        """
        device = "cuda:0"
        N1, M1 = 8, 5
        N2, M2 = 16, 11
        a = wp.zeros((N1, M1), dtype=float, device=device)
        b = wp.zeros((N2, M2), dtype=float, device=device)
        out = wp.zeros(2, dtype=int, device=device)

        _run_specialized(
            view_inner_shape_caller, dim=N1, inputs=[a, b, out], device=device
        )

        wp.synchronize_device(device)
        self.assertEqual(int(out.numpy()[0]), M1, "first call saw wrong shape from view-result")
        self.assertEqual(int(out.numpy()[1]), M2, "second call saw wrong shape from view-result")

    def test_codegen_nested_func_variants(self):
        """Each wp.func is emitted ONCE as a template; nested calls deduce types."""
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

        # Both wp.funcs are emitted as function templates with their
        # array arg as a deduced typename parameter (the same template
        # serves array_t<T> and baked_array_t<T,...> call sites).
        self.assertRegex(
            source,
            r"template<typename array_t_arr>\s*\n\s*\n?\s*//[^\n]*\n\s*static CUDA_CALLABLE wp::float32 inner_add_0\(",
        )
        self.assertRegex(
            source,
            r"template<typename array_t_arr>\s*\n\s*\n?\s*//[^\n]*\n\s*static CUDA_CALLABLE wp::float32 outer_scale_0\(",
        )
        # outer body calls inner with no explicit template args (deduced
        # from the array arg type at the call site).
        self.assertIn("inner_add_0(var_arr, var_i)", source)
        # Kernel materializes ``var_a`` as baked_array_t and passes it
        # by value to outer_scale_0 (typename deduces to baked_array_t).
        self.assertIn("outer_scale_0(var_a, var_0, var_1)", source)
        # And the kernel constructs the baked array struct from the raw
        # __restrict__ pointer.
        self.assertRegex(source, r"wp::baked_array_t<wp::float32, a_ndim,[^>]*>\s+var_a\{var_a_data\}")

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

    def test_per_kernel_opt_out(self):
        """``@wp.kernel(specialize=False)`` must override the global flag
        and prevent specialization for that kernel only."""
        from warp._src.context import user_modules

        N = 64
        device = "cuda:0"
        x = wp.array(np.ones(N, dtype=np.float32), device=device)
        y = wp.array(np.ones(N, dtype=np.float32), device=device)

        # Sanity: global flag is on (set in setUp).
        self.assertTrue(wp.config.enable_kernel_specialize)

        # Launch the opt-out kernel under graph capture.  Even though
        # the spec path is globally enabled, this kernel must take the
        # generic launch — no ``_spec_`` module should appear for it.
        spec_before = {k for k in user_modules if "axpy_no_spec_spec_" in k}
        with wp.ScopedCapture(device=device) as cap:
            wp.launch(axpy_no_spec, dim=N, inputs=[y, x, 2.0], device=device)
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)
        spec_after = {k for k in user_modules if "axpy_no_spec_spec_" in k}
        self.assertEqual(spec_before, spec_after, "specialize=False kernel must not register a spec module")

        # Result still correct: 2.0 * 1.0 + 1.0 == 3.0.
        np.testing.assert_allclose(y.numpy(), 3.0, rtol=1e-5)

    def test_per_kernel_opt_out_with_global_off(self):
        """``specialize=False`` is a no-op when the global flag is also
        off — kernel just runs through the generic path."""
        wp.config.enable_kernel_specialize = False
        N = 64
        device = "cuda:0"
        x = wp.array(np.ones(N, dtype=np.float32), device=device)
        y = wp.array(np.ones(N, dtype=np.float32), device=device)

        with wp.ScopedCapture(device=device) as cap:
            wp.launch(axpy_no_spec, dim=N, inputs=[y, x, 2.0], device=device)
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)

        np.testing.assert_allclose(y.numpy(), 3.0, rtol=1e-5)

    def test_autograd_under_spec(self):
        """Forward+backward via wp.Tape works correctly under spec.

        The spec path applies only to the forward launch (the dispatch
        check has ``not adjoint``).  ``runtime.tape.record_launch``
        records the ORIGINAL kernel reference, so ``tape.backward()``
        replays through ``wp.launch(kernel, ..., adjoint=True)`` and
        skips the spec branch.  The backward runs at baseline speed on
        the original module (which has ``enable_backward=True`` by
        default).  Gradients are correct because the backward sees the
        same runtime ``shape`` / ``strides`` as the forward saw at
        capture time.

        Documents the contract: spec is a forward-only optimization
        right now.  AD users still get correct gradients without
        opting out — they just don't get spec speedup on backward.
        """
        N = 8
        device = "cuda:0"
        x_np = np.arange(N, dtype=np.float32)

        x = wp.array(x_np, dtype=float, device=device, requires_grad=True)
        out = wp.zeros(N, dtype=float, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(scale_for_ad, dim=N, inputs=[x, out, 3.0], device=device)

        out.grad.fill_(1.0)
        tape.backward()
        wp.synchronize_device(device)

        np.testing.assert_allclose(out.numpy(), 3.0 * x_np, rtol=1e-5)
        np.testing.assert_allclose(x.grad.numpy(), 3.0 * np.ones(N, dtype=np.float32), rtol=1e-5)

    def test_eager_mode_warning_fires_once(self):
        """Spec launches outside graph capture should warn (once per
        process) — eager mode is correct but ~47% slower, so the user
        deserves a heads-up.  Refusing to spec would break debuggability."""
        import io
        from contextlib import redirect_stdout

        from warp._src import context

        # Clear the one-shot flag so this test exercises the first-warn path.
        context._reset_spec_eager_warning()

        N = 64
        device = "cuda:0"
        x = wp.array(np.ones(N, dtype=np.float32), device=device)
        y = wp.array(np.ones(N, dtype=np.float32), device=device)

        # Eager launch (no ScopedCapture).
        buf = io.StringIO()
        with redirect_stdout(buf):
            wp.launch(axpy, dim=N, inputs=[y, x, 2.0], device=device)
            wp.synchronize_device(device)
        msg = buf.getvalue()
        self.assertIn("spec path outside graph capture", msg)
        self.assertIn("ScopedCapture", msg)

        # Second eager launch — warning must NOT fire again.
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            wp.launch(axpy, dim=N, inputs=[y, x, 2.0], device=device)
            wp.synchronize_device(device)
        self.assertNotIn("spec path outside graph capture", buf2.getvalue())

        # Restore the flag for any subsequent tests / processes.
        context._reset_spec_eager_warning()

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

        # Scalar baking flows through a *single* wp.func definition:
        # ``scale_func_0`` is emitted once (no separate templated
        # overload).  The kernel template has ``s`` as an NTTP; the
        # kernel body declares ``const float var_s = s;`` and passes
        # ``var_s`` as a runtime arg to ``scale_func_0``.  NVRTC
        # inlines ``scale_func_0`` and constant-propagates ``var_s``
        # through the inlined body.
        cu_path = _find_spec_cu("scalar_through_func")
        self.assertIsNotNone(cu_path)
        with open(cu_path) as f:
            source = f.read()
        # Single (un-templated) wp.func definition for scale_func_0.
        self.assertRegex(source, r"static CUDA_CALLABLE wp::float32 scale_func_0\(")
        # Kernel materializes ``var_s`` from the NTTP and passes it as a
        # normal runtime arg.
        self.assertIn("var_s = s;", source)
        self.assertIn("scale_func_0(var_3, var_s)", source)

        np.testing.assert_allclose(out.numpy(), a.numpy() * 5.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
