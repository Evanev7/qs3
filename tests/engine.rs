use qs3::{
    AppendBatch, Commit, DType, DecodeBatch, Engine, EngineConfig, EngineLayer, EngineTrait,
    KvLayout, qsfi,
};
use std::ffi::{CStr, c_char, c_void};
use std::{mem, ptr};

const CUDA_SUCCESS: i32 = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

unsafe extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DeviceBuffer<T> {
    fn new(len: usize) -> Self {
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .expect("device allocation size overflow");
        let mut ptr = ptr::null_mut();
        assert_cuda(
            unsafe { cudaMalloc(&mut ptr, bytes) },
            "allocate CUDA test buffer",
        );
        Self {
            ptr: ptr.cast(),
            len,
        }
    }

    fn from_slice(values: &[T]) -> Self
    where
        T: Copy,
    {
        let buffer = Self::new(values.len());
        buffer.copy_from_slice(values);
        buffer
    }

    fn as_device_ptr(&self) -> qsfi::DevicePtr {
        self.ptr.cast()
    }

    fn copy_from_slice(&self, values: &[T])
    where
        T: Copy,
    {
        assert_eq!(values.len(), self.len);
        assert_cuda(
            unsafe {
                cudaMemcpy(
                    self.as_device_ptr(),
                    values.as_ptr().cast(),
                    mem::size_of_val(values),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "copy CUDA test buffer to device",
        );
    }

    fn memset(&self, value: u8) {
        let bytes = self
            .len
            .checked_mul(mem::size_of::<T>())
            .expect("device memset size overflow");
        assert_cuda(
            unsafe { cudaMemset(self.as_device_ptr(), i32::from(value), bytes) },
            "memset CUDA test buffer",
        );
    }
}

impl DeviceBuffer<u16> {
    fn assert_zero(&self, what: &str) {
        let mut host = vec![u16::MAX; self.len];
        assert_cuda(
            unsafe {
                cudaMemcpy(
                    host.as_mut_ptr().cast(),
                    self.as_device_ptr(),
                    mem::size_of_val(host.as_slice()),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            },
            what,
        );
        assert!(
            host.iter().all(|value| *value == 0),
            "{what}: expected all output elements to be zero"
        );
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.as_device_ptr());
            }
        }
    }
}

fn cuda_error_string(err: i32) -> String {
    if err == CUDA_SUCCESS {
        return "cudaSuccess".to_owned();
    }
    let ptr = unsafe { cudaGetErrorString(err) };
    if ptr.is_null() {
        return format!("CUDA error {err}");
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn assert_cuda(err: i32, what: &str) {
    assert_eq!(
        err,
        CUDA_SUCCESS,
        "{what}: {} ({err})",
        cuda_error_string(err)
    );
}

fn cuda_device_available() -> bool {
    let mut device_count = 0;
    let err = unsafe { cudaGetDeviceCount(&mut device_count) };
    if err != CUDA_SUCCESS || device_count == 0 {
        eprintln!(
            "SKIP: no CUDA device available: {} ({err})",
            cuda_error_string(err)
        );
        return false;
    }
    assert_cuda(unsafe { cudaSetDevice(0) }, "set CUDA test device");
    true
}

fn tensor3(
    data: qsfi::DevicePtr,
    dtype: qsfi::DTypeRaw,
    n: usize,
    heads: u32,
    head_dim: u32,
) -> qsfi::TensorDesc {
    let heads = i64::from(heads);
    let head_dim = i64::from(head_dim);
    qsfi::TensorDesc {
        data,
        dtype,
        ndim: 3,
        shape: [n as i64, heads, head_dim, 0, 0],
        stride: [heads * head_dim, head_dim, 1, 0, 0],
    }
}

fn tiny_config() -> EngineConfig {
    EngineConfig {
        device_ordinal: 0,
        stream: std::ptr::null_mut(),
        num_layers: 1,
        max_live_requests: 4,
        max_batch_size: 3,
        max_seq_len: 8,
        max_pages: 8,
        page_size: 4,
        hidden_size: 128,
        intermediate_size: 0,
        vocab_size: 0,
        num_q_heads: 2,
        num_kv_heads: 2,
        head_dim: 64,
        activation_dtype: DType::F16,
        kv_dtype: DType::F16,
        kv_layout: KvLayout::NHD,
        rope_theta: 10000.0,
        rope_scale: 1.0,
        logits_soft_cap: 0.0,
        qsfi_float_workspace_bytes: 64 << 20,
        qsfi_int_workspace_bytes: 64 << 20,
        qsfi_host_int_workspace_bytes: 64 << 20,
    }
}

#[test]
fn append_and_decode_layer_execute() {
    if !cuda_device_available() {
        return;
    }

    let config = tiny_config();
    let request_id = 71;
    let append_q_elems = 3 * config.num_q_heads as usize * config.head_dim as usize;
    let append_kv_elems = 3 * config.num_kv_heads as usize * config.head_dim as usize;
    let decode_q_elems = config.num_q_heads as usize * config.head_dim as usize;
    let decode_kv_elems = config.num_kv_heads as usize * config.head_dim as usize;

    let append_q = DeviceBuffer::<u16>::new(append_q_elems);
    let append_k = DeviceBuffer::<u16>::new(append_kv_elems);
    let append_v = DeviceBuffer::<u16>::new(append_kv_elems);
    let append_o = DeviceBuffer::<u16>::new(append_q_elems);
    let decode_q = DeviceBuffer::<u16>::new(decode_q_elems);
    let decode_k = DeviceBuffer::<u16>::new(decode_kv_elems);
    let decode_v = DeviceBuffer::<u16>::new(decode_kv_elems);
    let decode_o = DeviceBuffer::<u16>::new(decode_q_elems);
    let append_rope = DeviceBuffer::from_slice(&[0_i32, 1, 2]);
    let decode_rope = DeviceBuffer::from_slice(&[3_i32]);

    append_q.memset(0);
    append_k.memset(0);
    append_v.memset(0);
    append_o.memset(0xA5);
    decode_q.memset(0);
    decode_k.memset(0);
    decode_v.memset(0);
    decode_o.memset(0xA5);

    let mut session = Engine::new(config).unwrap();
    session
        .begin_append(AppendBatch {
            request_ids: &[request_id],
            token_indptr: &[0, 3],
            tokens: &[10, 11, 12],
        })
        .unwrap();

    let append_layer = EngineLayer {
        layer_idx: 0,
        q: tensor3(
            append_q.as_device_ptr(),
            qsfi::DTYPE_F16,
            3,
            config.num_q_heads,
            config.head_dim,
        ),
        k: tensor3(
            append_k.as_device_ptr(),
            qsfi::DTYPE_F16,
            3,
            config.num_kv_heads,
            config.head_dim,
        ),
        v: tensor3(
            append_v.as_device_ptr(),
            qsfi::DTYPE_F16,
            3,
            config.num_kv_heads,
            config.head_dim,
        ),
        o: tensor3(
            append_o.as_device_ptr(),
            qsfi::DTYPE_F16,
            3,
            config.num_q_heads,
            config.head_dim,
        ),
        q_rope_offset: append_rope.as_device_ptr(),
        lse: ptr::null_mut(),
        q_scale: 0.0,
        k_scale: 0.0,
        v_scale: 0.0,
    };
    unsafe {
        session.append_layer(&append_layer).unwrap();
    }
    assert_cuda(unsafe { cudaDeviceSynchronize() }, "sync append layer");
    append_o.assert_zero("append layer zero output");
    session
        .commit_batch(Commit {
            accepted_token_counts: None,
        })
        .unwrap();

    let append_state = session.state().unwrap();
    assert_eq!(append_state.live_seq_lens, &[3]);

    session
        .begin_decode(DecodeBatch {
            request_ids: &[request_id],
            tokens: &[13],
        })
        .unwrap();
    let decode_layer = EngineLayer {
        layer_idx: 0,
        q: tensor3(
            decode_q.as_device_ptr(),
            qsfi::DTYPE_F16,
            1,
            config.num_q_heads,
            config.head_dim,
        ),
        k: tensor3(
            decode_k.as_device_ptr(),
            qsfi::DTYPE_F16,
            1,
            config.num_kv_heads,
            config.head_dim,
        ),
        v: tensor3(
            decode_v.as_device_ptr(),
            qsfi::DTYPE_F16,
            1,
            config.num_kv_heads,
            config.head_dim,
        ),
        o: tensor3(
            decode_o.as_device_ptr(),
            qsfi::DTYPE_F16,
            1,
            config.num_q_heads,
            config.head_dim,
        ),
        q_rope_offset: decode_rope.as_device_ptr(),
        lse: ptr::null_mut(),
        q_scale: 0.0,
        k_scale: 0.0,
        v_scale: 0.0,
    };
    unsafe {
        session.decode_layer(&decode_layer).unwrap();
    }
    assert_cuda(unsafe { cudaDeviceSynchronize() }, "sync decode layer");
    decode_o.assert_zero("decode layer zero output");
    session
        .commit_batch(Commit {
            accepted_token_counts: None,
        })
        .unwrap();

    let decode_state = session.state().unwrap();
    assert_eq!(decode_state.live_seq_lens, &[4]);
}
