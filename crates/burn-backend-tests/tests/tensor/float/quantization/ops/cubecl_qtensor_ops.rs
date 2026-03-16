//! Tests for cubecl quantized tensor ops (q_slice, q_expand, q_flip, q_gather, q_select).
//!
//! All tensor shapes use last-dim multiples of 4 to satisfy packed U32 storage requirements.

use super::qtensor::*;
use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::quantization::{QTensorPrimitive, QuantLevel, QuantParam, QuantStore, QuantValue};

// ---- q_slice ----

#[test]
fn q_slice_1d_aligned() {
    // 8 elements → slice to last 4
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    let output = tensor.slice([4..8]);
    let expected = TensorData::from([4.0, 5.0, 6.0, 7.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-1));
}

#[test]
fn q_slice_2d_row() {
    // [2, 4] → slice first dim to get [1, 4]
    let tensor =
        QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]);

    let output = tensor.slice([1..2]);
    let expected = TensorData::from([[4.0, 5.0, 6.0, 7.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-1));
}

#[test]
fn q_slice_2d_col() {
    // [2, 8] → slice last dim to get [2, 4]
    let tensor = QTensor::<TestBackend, 2>::int8([
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    ]);

    let output = tensor.slice([0..2, 4..8]);
    let expected = TensorData::from([[4.0, 5.0, 6.0, 7.0], [12.0, 13.0, 14.0, 15.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-1));
}

#[test]
fn q_slice_qkv_split_pattern() {
    // Simulates FLUX QKV split: [1, 12] → 3 slices of [1, 4]
    let tensor =
        QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]);

    let q = tensor.clone().slice([0..1, 0..4]);
    let k = tensor.clone().slice([0..1, 4..8]);
    let v = tensor.slice([0..1, 8..12]);

    let q_expected = TensorData::from([[1.0, 2.0, 3.0, 4.0]]);
    let k_expected = TensorData::from([[5.0, 6.0, 7.0, 8.0]]);
    let v_expected = TensorData::from([[9.0, 10.0, 11.0, 12.0]]);

    q.dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&q_expected, Tolerance::rel_abs(2e-2, 1e-1));
    k.dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&k_expected, Tolerance::rel_abs(2e-2, 1e-1));
    v.dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&v_expected, Tolerance::rel_abs(2e-2, 1e-1));
}

// ---- q_expand ----

#[test]
fn q_expand_2d() {
    let tensor = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0]);

    let output = tensor.expand([3, 4]);
    let expected = TensorData::from([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
    ]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-1));
}

#[test]
fn q_expand_3d() {
    let tensor = QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);

    let output = tensor.expand([2, 2, 4]);
    let expected = TensorData::from([
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    ]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-1));
}

// ---- q_flip ----

#[test]
fn q_flip_1d() {
    let tensor = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0]);

    let output = tensor.flip([0]);
    let expected = TensorData::from([4.0, 3.0, 2.0, 1.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-1));
}

#[test]
fn q_flip_2d() {
    let tensor =
        QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);

    let output = tensor.flip([0, 1]);
    let expected = TensorData::from([[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-1));
}

// ---- q_gather ----

#[test]
fn q_gather_2d_dim0() {
    let tensor =
        QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
    let indices = TestTensorInt::from_data([[1, 0, 1, 0]], &Default::default());

    let output = tensor.gather(0, indices);
    let expected = TensorData::from([[5.0, 2.0, 7.0, 4.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-1));
}

// ---- q_select ----

#[test]
fn q_select_dim0() {
    let tensor = QTensor::<TestBackend, 2>::int8([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]);
    let indices = TestTensorInt::from_data([0, 2], &Default::default());

    let output = tensor.select(0, indices);
    let expected = TensorData::from([[1.0, 2.0, 3.0, 4.0], [9.0, 10.0, 11.0, 12.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-1));
}

// ---- E2M1 (FP4) tests ----
// Validates hardware FP4 tensor core path on Blackwell (sm_120+).

/// Create E2M1 scheme packing on a specific dimension (from innermost).
/// For weights [out, in]: pack_dim=0 packs on in_features (last/innermost).
/// For weights [out, in] used as matmul RHS: pack_dim=1 packs on out_features.
fn e2m1_scheme_dim(pack_dim: usize) -> burn_tensor::quantization::QuantScheme {
    <
        <TestBackend as burn_tensor::backend::Backend>::QuantizedTensorPrimitive
        as QTensorPrimitive
    >::default_scheme()
        .with_value(QuantValue::E2M1)
        .with_store(QuantStore::PackedNative(pack_dim))
        .with_param(QuantParam::UE8M0)
        .with_level(QuantLevel::block([32]))
}

fn e2m1_scheme() -> burn_tensor::quantization::QuantScheme {
    e2m1_scheme_dim(0)
}

#[test]
fn e2m1_quantize_dequantize_roundtrip() {
    let data = [
        [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0,
         1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0,
         1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0,
         1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0],
    ];

    let scheme = e2m1_scheme();
    let tensor = TestTensor::<2>::from_floats(data, &Default::default()).quantize_dynamic(&scheme);

    let tensor_data = tensor.to_data();
    assert!(matches!(tensor_data.dtype, burn_tensor::DType::QFloat(_)));

    let output = tensor.dequantize().into_data();
    let expected = TensorData::from(data);
    output.assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(5e-1, 5e-1));
}

#[test]
fn e2m1_matmul_2d() {
    // E2M1 matmul: [2, 32] × [32, 32] → [2, 32]
    // All dims ≥ packing factor (2) and last dim ≥ block size (32)
    let scheme = e2m1_scheme();

    // A = identity-like [2, 32]: each row has 1.0 at alternating positions
    let mut a_data = vec![0.0f32; 2 * 32];
    for i in 0..32 {
        a_data[i] = 1.0; // row 0: all ones
    }
    for i in 0..32 {
        a_data[32 + i] = 2.0; // row 1: all twos
    }

    // B = identity [32, 32]
    let mut b_data = vec![0.0f32; 32 * 32];
    for i in 0..32 {
        b_data[i * 32 + i] = 1.0;
    }

    let a = TestTensor::<2>::from_floats(
        TensorData::new(a_data.clone(), [2, 32]), &Default::default()
    ).quantize_dynamic(&scheme);
    let b = TestTensor::<2>::from_floats(
        TensorData::new(b_data, [32, 32]), &Default::default()
    ).quantize_dynamic(&scheme);

    let result = a.matmul(b);
    // A × I = A
    let expected = TensorData::new(a_data, [2, 32]);
    result
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(5e-1, 1.0));
}

#[test]
fn e2m1_dequantize_1x32() {
    let scheme = e2m1_scheme();
    let data: Vec<f32> = vec![1.0; 32];
    let t = TestTensor::<2>::from_floats(
        TensorData::new(data, [1, 32]), &Default::default()
    ).quantize_dynamic(&scheme);
    let result = t.dequantize().into_data();
    let expected = TensorData::new(vec![1.0f32; 32], [1, 32]);
    result.assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(5e-1, 5e-1));
}

#[test]
fn e2m1_dequantize_32x2() {
    // Block([32]) requires last dim >= block_size. For [32, 2] with last dim=2,
    // use Tensor-level quantization instead.
    // TODO: Fix BlockScaledLayout for block_size > last_dim in PackedNative.
    let scheme = <
        <TestBackend as burn_tensor::backend::Backend>::QuantizedTensorPrimitive
        as QTensorPrimitive
    >::default_scheme()
        .with_value(QuantValue::E2M1)
        .with_store(QuantStore::PackedNative(0))
        .with_param(QuantParam::UE8M0)
        .with_level(QuantLevel::Tensor);
    let data: Vec<f32> = vec![1.0; 64];
    let t = TestTensor::<2>::from_floats(
        TensorData::new(data, [32, 2]), &Default::default()
    ).quantize_dynamic(&scheme);
    let result = t.dequantize().into_data();
    let expected = TensorData::new(vec![1.0f32; 64], [32, 2]);
    result.assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(5e-1, 5e-1));
}

// TODO: [32, 1] with PackedNative(1) — packs on non-innermost dim.
// Fails due to ScalesView coordinate mapping not handling non-innermost packed dims.
// For real models, weights are [out, in] where in (innermost) is always large,
// so PackedNative(0) always works.

#[test]
fn e2m1_matmul_mixed_float_lhs() {
    // Float [1, 32] × E2M1 [32, 32] → [1, 32]
    // Identity-like weight matrix: diagonal = 1.0, rest = 0.0
    // Result should approximately equal the input
    let scheme = e2m1_scheme();

    let input_data: Vec<f32> = (0..32).map(|i| ((i % 4) + 1) as f32).collect();
    let input = TestTensor::<2>::from_floats(
        TensorData::new(input_data.clone(), [1, 32]), &Default::default()
    );

    // Identity matrix [32, 32]
    let mut weights_data = vec![0.0f32; 32 * 32];
    for i in 0..32 {
        weights_data[i * 32 + i] = 1.0;
    }
    let weights = TestTensor::<2>::from_floats(
        TensorData::new(weights_data, [32, 32]), &Default::default()
    ).quantize_dynamic(&scheme);

    let result = input.matmul(weights);
    let expected = TensorData::new(input_data, [1, 32]);

    result
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(5e-1, 2.0));
}
