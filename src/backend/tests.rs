use super::{
    BF16, Bf16Heads, DMat, DVec, F32, FloatStorage, GdnConvState, GdnRecurrentState,
    GdnStateIndexPolicy, I32, RouterScore, Workspace, qscb, qscu,
    qsfi::{FusedAddRmsNormBf16, RmsNormBf16, RopeApplyBf16},
};
use crate::{
    QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS, QWEN36_FULL_ATTN_KV_HIDDEN,
    QWEN36_FULL_ATTN_Q_HEADS, QWEN36_FULL_ATTN_Q_HIDDEN, QWEN36_FULL_ATTN_ROTARY_DIM,
    QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_NUM_Q_HEADS,
    QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_VALUE_DIM, QWEN36_MOE_MAX_TOP_K,
    Status,
    ffi::{self, sys},
};

use std::ffi::c_void;
use std::ptr;

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
fn qscu_descriptor_builders_validate_shapes_and_modes() {
    let gate = bf16_mat(10, 2, 8);
    let up = bf16_mat(11, 2, 8);
    let out = bf16_mat(12, 2, 8);
    assert!(qscu::silu_and_mul_desc(gate, up, out).is_ok());
    let padded_gate = DMat::new(device_ptr(13), 2, 8, 16).unwrap();
    assert!(matches!(
        qscu::silu_and_mul_desc(padded_gate, up, out),
        Err(Status::InvalidArgument)
    ));

    assert!(
        qscu::qwen36_shared_expert_gate_add_desc(
            f32_mat(110, 2, 1),
            bf16_mat(111, 2, 8),
            bf16_mat(112, 2, 8),
        )
        .is_ok()
    );
    assert!(matches!(
        qscu::qwen36_shared_expert_gate_add_desc(
            f32_mat(116, 2, 2),
            bf16_mat(117, 2, 8),
            bf16_mat(118, 2, 8),
        ),
        Err(Status::InvalidArgument)
    ));

    assert!(
        qscu::qwen36_full_attention_output_gate_desc(
            bf16_mat(119, 2, QWEN36_FULL_ATTN_Q_HIDDEN),
            bf16_mat(120, 2, QWEN36_FULL_ATTN_Q_HIDDEN),
        )
        .is_ok()
    );
    assert!(matches!(
        qscu::qwen36_full_attention_output_gate_desc(
            bf16_mat(121, 2, QWEN36_FULL_ATTN_Q_HIDDEN - 1),
            bf16_mat(122, 2, QWEN36_FULL_ATTN_Q_HIDDEN - 1),
        ),
        Err(Status::InvalidArgument)
    ));
    assert!(matches!(
        qscu::qwen36_full_attention_output_gate_desc(
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
        qscu::qwen36_full_attention_output_gate_desc(
            bf16_mat(125, 2, QWEN36_FULL_ATTN_Q_HIDDEN),
            bf16_mat(126, 1, QWEN36_FULL_ATTN_Q_HIDDEN),
        ),
        Err(Status::InvalidArgument)
    ));

    let token_ids = i32_vec(14, 2);
    let embedding = bf16_mat(15, 128, 8);
    assert!(qscu::embedding_gather_desc(token_ids, embedding, out, None, false).is_ok());
    assert!(matches!(
        qscu::embedding_gather_desc(token_ids, embedding, bf16_mat(16, 2, 7), None, false,),
        Err(Status::InvalidArgument)
    ));

    let logits = f32_mat(17, 2, 128);
    assert!(qscu::validate_logits_soft_cap(logits, 30.0).is_ok());
    assert!(qscu::validate_logits_soft_cap(logits, f32::NEG_INFINITY).is_ok());
    assert!(matches!(
        qscu::validate_logits_soft_cap(logits, f32::NAN),
        Err(Status::InvalidArgument)
    ));

    assert!(qscu::greedy_argmax_desc(logits, token_ids).is_ok());
    assert!(matches!(
        qscu::greedy_argmax_desc(logits, i32_vec(18, 1)),
        Err(Status::InvalidArgument)
    ));
    let huge_vocab = DMat::contiguous(device_ptr(19), 1, i32::MAX as u32 + 1).unwrap();
    assert!(matches!(
        qscu::greedy_argmax_desc(huge_vocab, i32_vec(20, 1)),
        Err(Status::Unsupported)
    ));

    assert!(
        qscu::router_topk_desc(
            logits,
            DMat::<I32>::contiguous(device_ptr(21), 2, 4).unwrap(),
            f32_mat(22, 2, 4),
            RouterScore::Softmax,
            true,
            1.0,
        )
        .is_ok()
    );
    assert!(matches!(
        qscu::router_topk_desc(
            logits,
            DMat::<I32>::contiguous(device_ptr(26), 2, QWEN36_MOE_MAX_TOP_K + 1,).unwrap(),
            f32_mat(27, 2, QWEN36_MOE_MAX_TOP_K + 1),
            RouterScore::Softmax,
            true,
            1.0,
        ),
        Err(Status::Unsupported)
    ));
    assert!(matches!(
        qscu::router_topk_desc(
            logits,
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
fn qscb_linear_specializations_encode_the_output_type() {
    let x = DMat::<BF16>::new(device_ptr(30), 4, 128, 160).unwrap();
    let weight = DMat::new(device_ptr(31), 256, 128, 128).unwrap();
    let bf16_out = DMat::<BF16>::new(device_ptr(32), 4, 256, 320).unwrap();
    let f32_out = DMat::<F32>::new(device_ptr(33), 4, 256, 320).unwrap();

    let bf16 = qscb::linear_desc(x, weight, bf16_out, Workspace::none()).unwrap();
    assert_eq!(bf16.out.dtype, ffi::DTYPE_BF16);
    assert_eq!(bf16.out.shape, [4, 256]);

    let f32 = qscb::linear_desc(x, weight, f32_out, Workspace::none()).unwrap();
    assert_eq!(f32.out.dtype, ffi::DTYPE_F32);
    assert_eq!(f32.out.shape, [4, 256]);

    assert!(matches!(
        qscb::linear_desc(
            x,
            weight,
            DMat::<F32>::new(device_ptr(34), 4, 128, 128).unwrap(),
            Workspace::none(),
        ),
        Err(Status::InvalidArgument)
    ));
}

#[test]
fn qsfi_descriptor_builders_accept_padded_rows() {
    let x = DMat::new(device_ptr(30), 4, 128, 160).unwrap();
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
    assert!(FusedAddRmsNormBf16::new(x, residual, bf16_vec(39, 128), 1.0e-6).is_ok());
    assert!(
        FusedAddRmsNormBf16::qwen_decoder_norm(x, residual, bf16_vec(140, 128), 1.0e-6,).is_ok()
    );

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
    let post = qscu::qwen36_gdn_post_conv_prepare_desc(
        bf16_mat(60, tokens, QWEN36_GDN_PACKED_DIM),
        bf16_mat(61, tokens, QWEN36_GDN_NUM_V_HEADS),
        bf16_mat(62, tokens, QWEN36_GDN_NUM_V_HEADS),
        bf16_vec(63, QWEN36_GDN_NUM_V_HEADS),
        bf16_vec(64, QWEN36_GDN_NUM_V_HEADS),
        q_heads(65, tokens),
        k_heads(66, tokens),
        v_heads(67, tokens),
    )
    .unwrap();
    assert_eq!(post.apply_qk_l2norm, 0);
    assert_eq!(post.forget_gate_output, sys::QSCU_GDN_FORGET_LOG_DECAY);
    assert!(matches!(
        qscu::qwen36_gdn_post_conv_prepare_desc(
            bf16_mat(60, tokens, QWEN36_GDN_PACKED_DIM),
            bf16_mat(61, tokens, QWEN36_GDN_NUM_V_HEADS),
            bf16_mat(62, tokens, QWEN36_GDN_NUM_V_HEADS),
            bf16_vec(63, QWEN36_GDN_NUM_V_HEADS),
            bf16_vec(64, QWEN36_GDN_NUM_V_HEADS),
            heads(69, tokens, 8, QWEN36_GDN_KEY_DIM),
            k_heads(66, tokens),
            v_heads(67, tokens),
        ),
        Err(Status::InvalidArgument)
    ));

    let gated = qscu::qwen36_gdn_gated_rmsnorm_desc(
        v_heads(70, tokens),
        v_heads(71, tokens),
        bf16_vec(72, QWEN36_GDN_VALUE_DIM),
        v_heads(73, tokens),
        1.0e-6,
    )
    .unwrap();
    assert_eq!(gated.gate_activation, sys::QSCU_ACTIVATION_SILU);
    assert!(matches!(
        qscu::qwen36_gdn_gated_rmsnorm_desc(
            v_heads(70, tokens),
            heads(71, tokens, QWEN36_GDN_NUM_V_HEADS - 1, QWEN36_GDN_VALUE_DIM),
            bf16_vec(72, QWEN36_GDN_VALUE_DIM),
            v_heads(73, tokens),
            1.0e-6,
        ),
        Err(Status::InvalidArgument)
    ));

    let conv = qscu::qwen36_gdn_causal_conv1d_desc(
        bf16_mat(74, tokens, QWEN36_GDN_PACKED_DIM),
        bf16_mat(75, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_CONV_WIDTH),
        bf16_vec(76, QWEN36_GDN_PACKED_DIM),
        GdnConvState::contiguous(device_ptr(77), FloatStorage::F32, 3).unwrap(),
        Some(i32_vec(78, tokens)),
        None,
        None,
        bf16_mat(79, tokens, QWEN36_GDN_PACKED_DIM),
        tokens,
    )
    .unwrap();
    assert_eq!(conv.activation, sys::QSCU_ACTIVATION_SILU);
    assert_eq!(conv.update_state, 1);
}

#[test]
fn gdn_recurrent_descriptors_validate_decode_and_prefill_shapes() {
    let tokens = 3;
    let q = q_heads(90, tokens);
    let k = k_heads(91, tokens);
    let v = v_heads(92, tokens);
    let a = bf16_mat(93, tokens, QWEN36_GDN_NUM_V_HEADS);
    let b = bf16_mat(94, tokens, QWEN36_GDN_NUM_V_HEADS);
    let a_log = bf16_vec(95, QWEN36_GDN_NUM_V_HEADS);
    let dt_bias = bf16_vec(96, QWEN36_GDN_NUM_V_HEADS);
    let state = recurrent_state(97);
    let out = v_heads(99, tokens);
    let decode = qscu::qwen36_gdn_decode_desc(
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        i32_vec(98, tokens),
        None,
        out,
    )
    .unwrap();
    assert_eq!(decode.use_qk_l2norm, 1);
    assert_eq!(decode.disable_state_update, 0);
    assert_eq!(decode.scale, 1.0 / (QWEN36_GDN_KEY_DIM as f32).sqrt());

    assert!(matches!(
        qscu::qwen36_gdn_decode_desc(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            i32_vec(100, 2),
            None,
            out,
        ),
        Err(Status::InvalidArgument)
    ));

    assert!(
        qscu::qwen36_gdn_prefill_desc(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            i32_vec(101, 3),
            i32_vec(102, 2),
            Some(i32_vec(103, 2)),
            out,
            2,
        )
        .is_ok()
    );
    assert!(matches!(
        qscu::qwen36_gdn_prefill_desc(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            i32_vec(104, 2),
            i32_vec(102, 2),
            Some(i32_vec(103, 2)),
            out,
            2,
        ),
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
