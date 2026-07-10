use super::*;
use crate::{QWEN36_FULL_ATTN_KV_HEADS, QWEN36_FULL_ATTN_KV_HIDDEN, QWEN36_FULL_ATTN_Q_HEADS};

use std::ffi::c_void;

fn device_ptr(offset: usize) -> ffi::DevicePtr {
    (0x1000usize + offset) as *mut c_void
}

fn bf16_mat(offset: usize, rows: u32, cols: u32) -> DMat<BF16> {
    DMat::contiguous(device_ptr(offset), rows, cols).unwrap()
}

fn f32_mat(offset: usize, rows: u32, cols: u32) -> DMat<F32> {
    DMat::contiguous(device_ptr(offset), rows, cols).unwrap()
}

fn bf16_vec(offset: usize, len: u32) -> DVec<BF16> {
    DVec::contiguous(device_ptr(offset), len).unwrap()
}

fn f32_vec(offset: usize, len: u32) -> DVec<F32> {
    DVec::contiguous(device_ptr(offset), len).unwrap()
}

fn i32_vec(offset: usize, len: u32) -> DVec<I32> {
    DVec::contiguous(device_ptr(offset), len).unwrap()
}

fn heads(offset: usize, tokens: u32, heads: u32, head_dim: u32) -> Bf16Heads {
    Bf16Heads::contiguous(device_ptr(offset), tokens, heads, head_dim).unwrap()
}

fn q_heads(offset: usize, tokens: u32) -> Bf16Heads {
    heads(offset, tokens, QWEN36_GDN_NUM_Q_HEADS, QWEN36_GDN_KEY_DIM)
}

fn k_heads(offset: usize, tokens: u32) -> Bf16Heads {
    heads(offset, tokens, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_KEY_DIM)
}

fn v_heads(offset: usize, tokens: u32) -> Bf16Heads {
    heads(offset, tokens, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_VALUE_DIM)
}

fn recurrent_state(offset: usize) -> GdnRecurrentState {
    GdnRecurrentState::contiguous(device_ptr(offset), FloatStorage::Bf16, 4).unwrap()
}

#[test]
fn generic_vec_and_mat_aliases_preserve_tensor_metadata() {
    let bf16_vec = bf16_vec(1, 4).tensor();
    assert_eq!(bf16_vec.dtype, ffi::DTYPE_BF16);
    assert_eq!(bf16_vec.shape, [4]);
    assert_eq!(bf16_vec.stride, [1]);

    let f32_vec = f32_vec(2, 5).tensor();
    assert_eq!(f32_vec.dtype, ffi::DTYPE_F32);

    let i32_vec = i32_vec(3, 6).tensor();
    assert_eq!(i32_vec.dtype, ffi::DTYPE_I32);

    let bf16_mat = bf16_mat(4, 2, 3);
    let f32_mat = DMat::<F32>::new(device_ptr(5), 2, 3, 8).unwrap();
    assert!(bf16_mat.same_shape(f32_mat));

    let f32_tensor = f32_mat.tensor();
    assert_eq!(f32_tensor.dtype, ffi::DTYPE_F32);
    assert_eq!(f32_tensor.shape, [2, 3]);
    assert_eq!(f32_tensor.stride, [8, 1]);

    let i32_tensor = DMat::<I32>::contiguous(device_ptr(6), 3, 2)
        .unwrap()
        .tensor();
    assert_eq!(i32_tensor.dtype, ffi::DTYPE_I32);
}

#[test]
fn handles_reject_null_zero_and_bad_strides() {
    assert!(matches!(
        DMat::<BF16>::contiguous(ptr::null_mut(), 1, 1),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        DMat::<BF16>::contiguous(device_ptr(1), 0, 1),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        DMat::<BF16>::new(device_ptr(2), 2, 4, 3),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        Bf16Heads::new(device_ptr(3), 2, 4, 64, 255, 64),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        Workspace::new(ptr::null_mut(), 1),
        Err(Status::InvalidArgument)
    ));
}

#[test]
fn rmsnorm_builder_supports_qwen_qk_flattened_per_head_norm() {
    let rows = 2;
    let heads = 16;
    let head_dim = 256;
    let flat_rows = rows * heads;
    let x = bf16_mat(130, flat_rows, head_dim);
    let weight = bf16_vec(131, head_dim);
    let out = bf16_mat(132, flat_rows, head_dim);

    let default_qwen = RmsNormBf16::new(x, weight, out, 1.0e-6).unwrap();
    assert_eq!(default_qwen.raw.hidden_size, head_dim);

    let qk = RmsNormBf16::qwen_qk_norm(x, weight, out, 1.0e-6).unwrap();
    assert_eq!(qk.raw.x.shape, [i64::from(flat_rows), i64::from(head_dim)]);
    assert_eq!(qk.raw.weight.shape, [i64::from(head_dim)]);

    let decoder = RmsNormBf16::qwen_decoder_norm(x, weight, out, 1.0e-6).unwrap();
    assert_eq!(decoder.raw.hidden_size, head_dim);

    let inplace = RmsNormBf16::qwen_qk_norm(x, weight, x, 1.0e-6).unwrap();
    assert_eq!(inplace.raw.out.data, inplace.raw.x.data);
    assert_eq!(inplace.raw.out.stride, inplace.raw.x.stride);
}

#[test]
fn qscu_descriptor_builders_validate_shapes_and_modes() {
    let gate = bf16_mat(10, 2, 8);
    let up = bf16_mat(11, 2, 8);
    let out = bf16_mat(12, 2, 8);
    assert!(SiluAndMulBf16::new(gate, up, out).is_ok());
    let padded_gate = DMat::new(device_ptr(13), 2, 8, 16).unwrap();
    assert!(matches!(
        SiluAndMulBf16::new(padded_gate, up, out),
        Err(Status::InvalidArgument)
    ));

    assert!(
        Qwen36SharedExpertGateAddBf16::new(
            Bf16OrF32Mat::F32(f32_mat(110, 2, 1)),
            bf16_mat(111, 2, 8),
            bf16_mat(112, 2, 8),
        )
        .is_ok()
    );
    assert!(
        Qwen36SharedExpertGateAddBf16::new(
            Bf16OrF32Mat::Bf16(bf16_mat(113, 2, 1)),
            bf16_mat(114, 2, 8),
            bf16_mat(115, 2, 8),
        )
        .is_ok()
    );
    assert!(matches!(
        Qwen36SharedExpertGateAddBf16::new(
            Bf16OrF32Mat::F32(f32_mat(116, 2, 2)),
            bf16_mat(117, 2, 8),
            bf16_mat(118, 2, 8),
        ),
        Err(Status::InvalidArgument)
    ));

    assert!(
        Qwen36FullAttentionOutputGateBf16::new(
            bf16_mat(119, 2, QWEN36_FULL_ATTN_Q_HIDDEN),
            bf16_mat(120, 2, QWEN36_FULL_ATTN_Q_HIDDEN),
        )
        .is_ok()
    );
    assert!(matches!(
        Qwen36FullAttentionOutputGateBf16::new(
            bf16_mat(121, 2, QWEN36_FULL_ATTN_Q_HIDDEN - 1),
            bf16_mat(122, 2, QWEN36_FULL_ATTN_Q_HIDDEN - 1),
        ),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        Qwen36FullAttentionOutputGateBf16::new(
            DMat::<BF16>::new(
                device_ptr(123),
                2,
                QWEN36_FULL_ATTN_Q_HIDDEN,
                QWEN36_FULL_ATTN_Q_HIDDEN + 8,
            )
            .unwrap(),
            bf16_mat(124, 2, QWEN36_FULL_ATTN_Q_HIDDEN),
        ),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        Qwen36FullAttentionOutputGateBf16::new(
            bf16_mat(125, 2, QWEN36_FULL_ATTN_Q_HIDDEN),
            bf16_mat(126, 1, QWEN36_FULL_ATTN_Q_HIDDEN),
        ),
        Err(Status::InvalidArgument)
    ));

    let token_ids = i32_vec(14, 2);
    let embedding = bf16_mat(15, 128, 8);
    assert!(EmbeddingGatherBf16::new(token_ids, embedding, out).is_ok());
    assert!(matches!(
        EmbeddingGatherBf16::new(token_ids, embedding, bf16_mat(16, 2, 7)),
        Err(Status::InvalidArgument)
    ));

    let logits = f32_mat(17, 2, 128);
    assert!(LogitsSoftCapF32::new(logits, 30.0).is_ok());
    assert!(LogitsSoftCapF32::new(logits, f32::NEG_INFINITY).is_ok());
    assert!(matches!(
        LogitsSoftCapF32::new(logits, f32::NAN),
        Err(Status::InvalidArgument)
    ));

    assert!(GreedyArgmaxF32::new(logits, token_ids).is_ok());
    assert!(matches!(
        GreedyArgmaxF32::new(logits, i32_vec(18, 1)),
        Err(Status::InvalidArgument)
    ));
    let huge_vocab = DMat::contiguous(device_ptr(19), 1, i32::MAX as u32 + 1).unwrap();
    assert!(matches!(
        GreedyArgmaxF32::new(huge_vocab, i32_vec(20, 1)),
        Err(Status::Unsupported)
    ));

    assert!(
        RouterTopK::new(
            Bf16OrF32Mat::F32(logits),
            DMat::<I32>::contiguous(device_ptr(21), 2, 4).unwrap(),
            f32_mat(22, 2, 4),
            RouterScore::Softmax,
            true,
            1.0,
        )
        .is_ok()
    );
    assert!(
        RouterTopK::new(
            Bf16OrF32Mat::Bf16(bf16_mat(23, 2, 128)),
            DMat::<I32>::contiguous(device_ptr(24), 2, 4).unwrap(),
            f32_mat(25, 2, 4),
            RouterScore::Sigmoid,
            false,
            0.5,
        )
        .is_ok()
    );
    assert!(matches!(
        RouterTopK::new(
            Bf16OrF32Mat::F32(logits),
            DMat::<I32>::contiguous(device_ptr(26), 2, QWEN36_MOE_MAX_TOP_K + 1,).unwrap(),
            f32_mat(27, 2, QWEN36_MOE_MAX_TOP_K + 1),
            RouterScore::Softmax,
            true,
            1.0,
        ),
        Err(Status::Unsupported)
    ));
    assert!(matches!(
        RouterTopK::new(
            Bf16OrF32Mat::F32(logits),
            DMat::<I32>::contiguous(device_ptr(28), 2, 4).unwrap(),
            f32_mat(29, 2, 4),
            RouterScore::Softmax,
            true,
            0.0,
        ),
        Err(Status::InvalidArgument)
    ));
}

#[test]
fn qscb_and_qsfi_descriptor_builders_accept_padded_rows() {
    let x = DMat::new(device_ptr(30), 4, 128, 160).unwrap();
    let weight = DMat::new(device_ptr(31), 256, 128, 128).unwrap();
    let out = DMat::new(device_ptr(32), 4, 256, 320).unwrap();
    assert!(Bf16Gemm::new(x, weight, Bf16OrF32Mat::F32(out), Workspace::none()).is_ok());
    assert!(matches!(
        Bf16Gemm::new(
            x,
            weight,
            Bf16OrF32Mat::F32(DMat::new(device_ptr(33), 4, 128, 128).unwrap()),
            Workspace::none(),
        ),
        Err(Status::InvalidArgument)
    ));

    let norm_out = DMat::new(device_ptr(34), 4, 128, 160).unwrap();
    assert!(RmsNormBf16::new(x, bf16_vec(35, 128), norm_out, 1.0e-6).is_ok());
    assert!(matches!(
        RmsNormBf16::new(x, bf16_vec(36, 64), norm_out, 1.0e-6),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        RmsNormBf16::new(x, bf16_vec(37, 128), norm_out, 0.0),
        Err(Status::InvalidArgument)
    ));

    let residual = DMat::new(device_ptr(38), 4, 128, 192).unwrap();
    let fused = FusedAddRmsNormBf16::new(x, residual, bf16_vec(39, 128), 1.0e-6).unwrap();
    assert_eq!(fused.raw.out.data, fused.raw.x.data);
    let qwen_fused =
        FusedAddRmsNormBf16::qwen_decoder_norm(x, residual, bf16_vec(140, 128), 1.0e-6).unwrap();
    assert_eq!(qwen_fused.raw.out.data, qwen_fused.raw.x.data);

    let q = Bf16Heads::new(device_ptr(40), 2, 4, 128, 1024, 128).unwrap();
    let k = Bf16Heads::new(device_ptr(41), 2, 2, 128, 512, 128).unwrap();
    let q_out = Bf16Heads::new(device_ptr(42), 2, 4, 128, 1024, 128).unwrap();
    let k_out = Bf16Heads::new(device_ptr(43), 2, 2, 128, 512, 128).unwrap();
    assert!(RopeApplyBf16::new(q, k, q_out, k_out, i32_vec(44, 2), 128).is_ok());
    assert!(matches!(
        RopeApplyBf16::new(
            heads(45, 2, 4, 96),
            heads(46, 2, 2, 96),
            heads(47, 2, 4, 96),
            heads(48, 2, 2, 96),
            i32_vec(49, 2),
            96,
        ),
        Err(Status::Unsupported)
    ));
    assert!(matches!(
        RopeApplyBf16::new(q, k, q_out, k_out, i32_vec(50, 2), 0),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        RopeApplyBf16::new(q, k, q_out, k_out, i32_vec(51, 2), 127),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        RopeApplyBf16::new(q, k, q_out, k_out, i32_vec(52, 2), 256),
        Err(Status::InvalidArgument)
    ));
    let qwen36_q = Bf16Heads::new(
        device_ptr(53),
        2,
        QWEN36_FULL_ATTN_Q_HEADS,
        QWEN36_FULL_ATTN_HEAD_DIM,
        QWEN36_FULL_ATTN_Q_HIDDEN,
        QWEN36_FULL_ATTN_HEAD_DIM,
    )
    .unwrap();
    let qwen36_k = Bf16Heads::new(
        device_ptr(54),
        2,
        QWEN36_FULL_ATTN_KV_HEADS,
        QWEN36_FULL_ATTN_HEAD_DIM,
        QWEN36_FULL_ATTN_KV_HIDDEN,
        QWEN36_FULL_ATTN_HEAD_DIM,
    )
    .unwrap();
    let qwen36_q_out = Bf16Heads::new(
        device_ptr(55),
        2,
        QWEN36_FULL_ATTN_Q_HEADS,
        QWEN36_FULL_ATTN_HEAD_DIM,
        QWEN36_FULL_ATTN_Q_HIDDEN,
        QWEN36_FULL_ATTN_HEAD_DIM,
    )
    .unwrap();
    let qwen36_k_out = Bf16Heads::new(
        device_ptr(56),
        2,
        QWEN36_FULL_ATTN_KV_HEADS,
        QWEN36_FULL_ATTN_HEAD_DIM,
        QWEN36_FULL_ATTN_KV_HIDDEN,
        QWEN36_FULL_ATTN_HEAD_DIM,
    )
    .unwrap();
    assert!(
        RopeApplyBf16::new(
            qwen36_q,
            qwen36_k,
            qwen36_q_out,
            qwen36_k_out,
            i32_vec(57, 2),
            QWEN36_FULL_ATTN_ROTARY_DIM,
        )
        .is_ok()
    );
    assert!(matches!(
        RopeApplyBf16::new(
            qwen36_q,
            qwen36_k,
            qwen36_q_out,
            qwen36_k_out,
            i32_vec(58, 2),
            128,
        ),
        Err(Status::Unsupported)
    ));
    let alias_bad_stride = Bf16Heads::contiguous(device_ptr(40), 2, 4, 128).unwrap();
    assert!(matches!(
        RopeApplyBf16::new(q, k, alias_bad_stride, k_out, i32_vec(59, 2), 128),
        Err(Status::InvalidArgument)
    ));
}

#[test]
fn gdn_prep_descriptors_enforce_qwen36_shapes() {
    let tokens = 2;
    let post = GdnPostConvPrepareBf16Args {
        conv_out: bf16_mat(60, tokens, QWEN36_GDN_PACKED_DIM),
        a: bf16_mat(61, tokens, QWEN36_GDN_NUM_V_HEADS),
        b: bf16_mat(62, tokens, QWEN36_GDN_NUM_V_HEADS),
        a_log: bf16_vec(63, QWEN36_GDN_NUM_V_HEADS),
        dt_bias: bf16_vec(64, QWEN36_GDN_NUM_V_HEADS),
        q: q_heads(65, tokens),
        k: k_heads(66, tokens),
        v: v_heads(67, tokens),
        g_out: Some(f32_mat(68, tokens, QWEN36_GDN_NUM_V_HEADS)),
        beta_out: None,
        apply_qk_l2norm: true,
        l2norm_eps: 1.0e-6,
        forget_gate_output: GdnForgetGateOutput::LinearAlpha,
    };
    assert!(GdnPostConvPrepareBf16::new(post).is_ok());

    let mut bad_post = post;
    bad_post.q = heads(69, tokens, 8, QWEN36_GDN_KEY_DIM);
    assert!(matches!(
        GdnPostConvPrepareBf16::new(bad_post),
        Err(Status::InvalidArgument)
    ));

    let gated = GdnRmsNormGatedBf16Args {
        x: v_heads(70, tokens),
        gate: v_heads(71, tokens),
        weight: Bf16OrF32Vec::F32(f32_vec(72, QWEN36_GDN_VALUE_DIM)),
        out: v_heads(73, tokens),
        eps: 1.0e-6,
        gate_activation: Activation::Silu,
    };
    assert!(GdnRmsNormGatedBf16::new(gated).is_ok());
    let mut bad_gated = gated;
    bad_gated.gate_activation = Activation::None;
    assert!(matches!(
        GdnRmsNormGatedBf16::new(bad_gated),
        Err(Status::Unsupported)
    ));

    let conv = GdnCausalConv1dBf16Args {
        x: bf16_mat(74, tokens, QWEN36_GDN_PACKED_DIM),
        weight: bf16_mat(75, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_CONV_WIDTH),
        bias: Some(Bf16OrF32Vec::Bf16(bf16_vec(76, QWEN36_GDN_PACKED_DIM))),
        state: GdnConvState::contiguous(device_ptr(77), FloatStorage::F32, 3).unwrap(),
        state_read_indices: Some(i32_vec(78, tokens)),
        state_write_indices: None,
        seq_indptr: None,
        out: bf16_mat(79, tokens, QWEN36_GDN_PACKED_DIM),
        batch_size: tokens,
        activation: Activation::Silu,
        update_state: true,
    };
    assert!(GdnCausalConv1dBf16::new(conv).is_ok());
    let mut bad_conv = conv;
    bad_conv.activation = Activation::Sigmoid;
    assert!(matches!(
        GdnCausalConv1dBf16::new(bad_conv),
        Err(Status::Unsupported)
    ));
}

#[test]
fn gdn_recurrent_descriptors_validate_decode_and_prefill_shapes() {
    let tokens = 3;
    let decode = GdnDecodeBf16Args {
        q: q_heads(90, tokens),
        k: k_heads(91, tokens),
        v: v_heads(92, tokens),
        a: bf16_mat(93, tokens, QWEN36_GDN_NUM_V_HEADS),
        b: bf16_mat(94, tokens, QWEN36_GDN_NUM_V_HEADS),
        a_log: bf16_vec(95, QWEN36_GDN_NUM_V_HEADS),
        dt_bias: bf16_vec(96, QWEN36_GDN_NUM_V_HEADS),
        state: recurrent_state(97),
        state_indices: i32_vec(98, tokens),
        state_out_indices: None,
        out: v_heads(99, tokens),
        scale: 0.08838835,
        use_qk_l2norm: true,
        disable_state_update: false,
    };
    assert!(GdnDecodeBf16::new(decode).is_ok());

    let mut bad_decode = decode;
    bad_decode.scale = 0.0;
    assert!(matches!(
        GdnDecodeBf16::new(bad_decode),
        Err(Status::InvalidArgument)
    ));
    let mut bad_indices = decode;
    bad_indices.state_indices = i32_vec(100, 2);
    assert!(matches!(
        GdnDecodeBf16::new(bad_indices),
        Err(Status::InvalidArgument)
    ));

    let prefill = GdnPrefillBf16Args {
        q: decode.q,
        k: decode.k,
        v: decode.v,
        a: decode.a,
        b: decode.b,
        a_log: decode.a_log,
        dt_bias: decode.dt_bias,
        state: decode.state,
        seq_indptr: i32_vec(101, 3),
        state_indices: i32_vec(102, 2),
        state_out_indices: Some(i32_vec(103, 2)),
        out: decode.out,
        batch_size: 2,
        scale: decode.scale,
        use_qk_l2norm: decode.use_qk_l2norm,
        disable_state_update: decode.disable_state_update,
    };
    assert!(GdnPrefillBf16::new(prefill).is_ok());
    let mut bad_prefill = prefill;
    bad_prefill.seq_indptr = i32_vec(104, 2);
    assert!(matches!(
        GdnPrefillBf16::new(bad_prefill),
        Err(Status::InvalidArgument)
    ));
}

#[test]
fn gdn_state_index_policy_validates_host_indices() {
    assert_eq!(
        GdnStateIndexPolicy::NegativeSkips.validate_host_indices(&[-1, 0, 3], 4),
        Ok(())
    );
    assert_eq!(
        GdnStateIndexPolicy::NonNegative.validate_host_indices(&[-1, 0], 4),
        Err(Status::InvalidArgument)
    );
    assert_eq!(
        GdnStateIndexPolicy::NegativeSkips.validate_host_indices(&[4], 4),
        Err(Status::InvalidArgument)
    );
    assert_eq!(
        GdnStateIndexPolicy::NegativeSkips.validate_host_indices(&[0], 0),
        Err(Status::InvalidArgument)
    );
}
