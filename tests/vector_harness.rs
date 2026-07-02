#![allow(dead_code)]

use qs3::ffi;

use std::{
    collections::{BTreeMap, HashSet},
    ffi::{CStr, c_char, c_void},
    fmt, fs, io, mem, path,
    path::{Path, PathBuf},
    ptr,
};

const DEFAULT_VECTOR_ROOT: &str = "build/vectors/qwen36_semantics";
const QSFI_STATUS_OK: ffi::StatusRaw = 0;
const CUDA_SUCCESS: i32 = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const BF16_NORM_ABS_TOL: f32 = 0.018;
const BF16_RESIDUAL_ABS_TOL: f32 = 0.012;
const BF16_GDN_CONV_ABS_TOL: f32 = 0.04;
const BF16_GDN_NORM_ABS_TOL: f32 = 0.04;
const BF16_GDN_RECURRENCE_OUTPUT_ABS_TOL: f32 = 0.06;
const BF16_GDN_RECURRENCE_STATE_ABS_TOL: f32 = 0.08;
const GDN_F32_ABS_TOL: f32 = 1.0e-4;
const QSCU_ACTIVATION_SILU: u32 = 1;
const QSCU_GDN_FORGET_LOG_DECAY: u32 = 0;
const QSCU_GDN_FORGET_LINEAR_ALPHA: u32 = 1;
const QSCU_GDN_STATE_LAYOUT_VK: u32 = 0;

unsafe extern "C" {
    fn cudaGetErrorString(error: i32) -> *const c_char;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn qsfi_status_string(status: ffi::StatusRaw) -> *const c_char;
    fn qsfi_context_create(desc: *const QsfiContextDesc, out: *mut *mut c_void) -> ffi::StatusRaw;
    fn qsfi_context_destroy(ctx: *mut c_void);
    fn qsfi_rmsnorm(ctx: *mut c_void, desc: *const ffi::RmsnormDesc) -> ffi::StatusRaw;
    fn qsfi_fused_add_rmsnorm(
        ctx: *mut c_void,
        desc: *const ffi::FusedAddRmsnormDesc,
    ) -> ffi::StatusRaw;
    fn qscu_qwen36_gdn_causal_conv1d_bf16(
        desc: *const QscuQwen36GdnCausalConv1dDesc,
        stream: ffi::CudaStream,
    ) -> ffi::StatusRaw;
    fn qscu_qwen36_gdn_post_conv_prepare_bf16(
        desc: *const QscuQwen36GdnPostConvPrepareDesc,
        stream: ffi::CudaStream,
    ) -> ffi::StatusRaw;
    fn qscu_qwen36_gdn_rmsnorm_gated_bf16(
        desc: *const QscuQwen36GdnRmsnormGatedDesc,
        stream: ffi::CudaStream,
    ) -> ffi::StatusRaw;
    fn qscu_gdn_decode(ctx: *mut c_void, desc: *const QscuGdnDecodeDesc) -> ffi::StatusRaw;
    fn qscu_gdn_prefill(ctx: *mut c_void, desc: *const QscuGdnPrefillDesc) -> ffi::StatusRaw;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QscuQwen36GdnCausalConv1dDesc {
    x: ffi::Tensor2,
    weight: ffi::Tensor2,
    bias: ffi::Tensor1,
    state: ffi::Tensor3,
    state_read_indices: ffi::Tensor1,
    state_write_indices: ffi::Tensor1,
    seq_indptr: ffi::DevicePtr,
    out: ffi::Tensor2,
    num_tokens: u32,
    batch_size: u32,
    activation: u32,
    update_state: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QscuQwen36GdnPostConvPrepareDesc {
    conv_out: ffi::Tensor2,
    a: ffi::Tensor2,
    b: ffi::Tensor2,
    a_log: ffi::Tensor1,
    dt_bias: ffi::Tensor1,
    q: ffi::Tensor3,
    k: ffi::Tensor3,
    v: ffi::Tensor3,
    g_out: ffi::Tensor2,
    beta_out: ffi::Tensor2,
    num_tokens: u32,
    apply_qk_l2norm: u32,
    l2norm_eps: f32,
    forget_gate_output: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QscuQwen36GdnRmsnormGatedDesc {
    x: ffi::Tensor3,
    gate: ffi::Tensor3,
    weight: ffi::Tensor1,
    out: ffi::Tensor3,
    num_tokens: u32,
    eps: f32,
    gate_activation: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QscuGdnPrefillDesc {
    q: ffi::Tensor3,
    k: ffi::Tensor3,
    v: ffi::Tensor3,
    a: ffi::Tensor2,
    b: ffi::Tensor2,
    a_log: ffi::Tensor1,
    dt_bias: ffi::Tensor1,
    state: ffi::Tensor4,
    seq_indptr: ffi::DevicePtr,
    state_indices: ffi::Tensor1,
    state_out_indices: ffi::Tensor1,
    out: ffi::Tensor3,
    batch_size: u32,
    total_tokens: u32,
    num_q_heads: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    key_dim: u32,
    value_dim: u32,
    state_layout: u32,
    scale: f32,
    use_qk_l2norm: u32,
    disable_state_update: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QscuGdnDecodeDesc {
    q: ffi::Tensor3,
    k: ffi::Tensor3,
    v: ffi::Tensor3,
    a: ffi::Tensor2,
    b: ffi::Tensor2,
    a_log: ffi::Tensor1,
    dt_bias: ffi::Tensor1,
    state: ffi::Tensor4,
    state_indices: ffi::Tensor1,
    state_out_indices: ffi::Tensor1,
    out: ffi::Tensor3,
    num_tokens: u32,
    num_q_heads: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    key_dim: u32,
    value_dim: u32,
    state_layout: u32,
    scale: f32,
    use_qk_l2norm: u32,
    disable_state_update: u32,
}

#[derive(Debug)]
struct VectorError {
    message: String,
}

type VectorResult<T> = Result<T, VectorError>;

impl VectorError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for VectorError {}

impl From<io::Error> for VectorError {
    fn from(err: io::Error) -> Self {
        Self::new(err.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VectorDType {
    Bf16,
    F32,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}

impl VectorDType {
    fn parse(value: &str) -> VectorResult<Self> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "i8" => Ok(Self::I8),
            "u8" => Ok(Self::U8),
            "i16" => Ok(Self::I16),
            "u16" => Ok(Self::U16),
            "i32" => Ok(Self::I32),
            "u32" => Ok(Self::U32),
            "i64" => Ok(Self::I64),
            "u64" => Ok(Self::U64),
            _ => Err(VectorError::new(format!("unsupported dtype {value:?}"))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
        }
    }

    fn byte_width(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::Bf16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    fn ffi_dtype(self) -> Option<ffi::DTypeRaw> {
        match self {
            Self::Bf16 => Some(ffi::DTYPE_BF16),
            Self::F32 => Some(ffi::DTYPE_F32),
            Self::I32 => Some(ffi::DTYPE_I32),
            Self::I8 => Some(ffi::DTYPE_I8),
            Self::U8 => Some(ffi::DTYPE_U8),
            Self::U32 => Some(ffi::DTYPE_U32),
            Self::F64 | Self::I16 | Self::U16 | Self::I64 | Self::U64 => None,
        }
    }
}

#[derive(Clone, Debug)]
struct TensorSpec {
    name: String,
    dtype: VectorDType,
    shape: Vec<usize>,
    file: String,
    role: Option<String>,
}

impl TensorSpec {
    fn element_count(&self) -> VectorResult<usize> {
        self.shape.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                VectorError::new(format!("tensor {} element count overflows", self.name))
            })
        })
    }

    fn byte_len(&self) -> VectorResult<usize> {
        self.element_count()?
            .checked_mul(self.dtype.byte_width())
            .ok_or_else(|| VectorError::new(format!("tensor {} byte length overflows", self.name)))
    }
}

#[derive(Debug)]
struct VectorManifest {
    name: String,
    groups: Vec<String>,
    tensors: Vec<TensorSpec>,
}

#[derive(Debug)]
struct RawTensor {
    spec: TensorSpec,
    bytes: Vec<u8>,
}

impl RawTensor {
    fn len(&self) -> usize {
        self.spec.element_count().expect("validated tensor shape")
    }

    fn require_dtype(&self, expected: VectorDType) -> VectorResult<()> {
        if self.spec.dtype == expected {
            Ok(())
        } else {
            Err(VectorError::new(format!(
                "tensor {} has dtype {}, expected {}",
                self.spec.name,
                self.spec.dtype.name(),
                expected.name()
            )))
        }
    }

    fn as_bf16_words(&self) -> VectorResult<Vec<u16>> {
        self.require_dtype(VectorDType::Bf16)?;
        self.read_u16_words()
    }

    fn as_u16_words(&self) -> VectorResult<Vec<u16>> {
        self.require_dtype(VectorDType::U16)?;
        self.read_u16_words()
    }

    fn as_f32_vec(&self) -> VectorResult<Vec<f32>> {
        self.require_dtype(VectorDType::F32)?;
        self.bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                Ok(f32::from_le_bytes(bytes))
            })
            .collect()
    }

    fn as_i32_vec(&self) -> VectorResult<Vec<i32>> {
        self.require_dtype(VectorDType::I32)?;
        self.bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                Ok(i32::from_le_bytes(bytes))
            })
            .collect()
    }

    fn read_u16_words(&self) -> VectorResult<Vec<u16>> {
        self.bytes
            .chunks_exact(2)
            .map(|chunk| Ok(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()
    }
}

#[derive(Debug)]
struct VectorArtifact {
    manifest_path: PathBuf,
    manifest: VectorManifest,
    tensors: BTreeMap<String, RawTensor>,
}

#[derive(Debug)]
struct NormMismatchMetadata {
    standard_rmsnorm_expected_fail: bool,
    standard_rmsnorm_bf16_mismatch_count: usize,
}

impl VectorArtifact {
    fn load(manifest_path: impl AsRef<Path>) -> VectorResult<Self> {
        let manifest_path = manifest_path.as_ref().to_owned();
        let text = fs::read_to_string(&manifest_path)
            .map_err(|err| VectorError::new(format!("read {}: {err}", manifest_path.display())))?;
        let value = JsonParser::new(&text).parse()?;
        let manifest = parse_manifest(&value)?;
        let base_dir = manifest_path
            .parent()
            .ok_or_else(|| VectorError::new("manifest path has no parent"))?;

        let mut tensors = BTreeMap::new();
        for spec in &manifest.tensors {
            let tensor_path = base_dir.join(&spec.file);
            let bytes = fs::read(&tensor_path).map_err(|err| {
                VectorError::new(format!("read tensor {}: {err}", tensor_path.display()))
            })?;
            let expected_len = spec.byte_len()?;
            if bytes.len() != expected_len {
                return Err(VectorError::new(format!(
                    "tensor {} byte length mismatch: file has {}, manifest expects {}",
                    spec.name,
                    bytes.len(),
                    expected_len
                )));
            }
            tensors.insert(
                spec.name.clone(),
                RawTensor {
                    spec: spec.clone(),
                    bytes,
                },
            );
        }

        Ok(Self {
            manifest_path,
            manifest,
            tensors,
        })
    }

    fn tensor(&self, name: &str) -> VectorResult<&RawTensor> {
        self.tensors.get(name).ok_or_else(|| {
            VectorError::new(format!(
                "{} is missing tensor {name:?}",
                self.manifest_path.display()
            ))
        })
    }
}

struct DeviceTensor<T> {
    ptr: *mut T,
    len: usize,
    shape: Vec<usize>,
    dtype: Option<ffi::DTypeRaw>,
}

impl DeviceTensor<u16> {
    fn from_bf16(tensor: &RawTensor) -> VectorResult<Self> {
        Self::from_slice_with_dtype(
            &tensor.as_bf16_words()?,
            tensor.spec.shape.clone(),
            Some(ffi::DTYPE_BF16),
        )
    }

    fn zeroed_bf16(shape: Vec<usize>) -> VectorResult<Self> {
        let len = shape.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| VectorError::new("device tensor element count overflows"))
        })?;
        Self::from_slice_with_dtype(&vec![0u16; len], shape, Some(ffi::DTYPE_BF16))
    }

    fn from_u16(tensor: &RawTensor) -> VectorResult<Self> {
        Self::from_slice_with_dtype(&tensor.as_u16_words()?, tensor.spec.shape.clone(), None)
    }

    fn to_u16_vec(&self, what: &str) -> Vec<u16> {
        let mut host = vec![0u16; self.len];
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
        host
    }
}

impl DeviceTensor<f32> {
    fn from_f32(tensor: &RawTensor) -> VectorResult<Self> {
        Self::from_slice_with_dtype(
            &tensor.as_f32_vec()?,
            tensor.spec.shape.clone(),
            Some(ffi::DTYPE_F32),
        )
    }

    fn zeroed_f32(shape: Vec<usize>) -> VectorResult<Self> {
        let len = shape.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| VectorError::new("device tensor element count overflows"))
        })?;
        Self::from_slice_with_dtype(&vec![0.0f32; len], shape, Some(ffi::DTYPE_F32))
    }

    fn to_f32_vec(&self, what: &str) -> Vec<f32> {
        let mut host = vec![0.0f32; self.len];
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
        host
    }
}

impl DeviceTensor<i32> {
    fn from_i32(tensor: &RawTensor) -> VectorResult<Self> {
        Self::from_slice_with_dtype(
            &tensor.as_i32_vec()?,
            tensor.spec.shape.clone(),
            Some(ffi::DTYPE_I32),
        )
    }

    fn from_i32_slice(values: &[i32], shape: Vec<usize>) -> VectorResult<Self> {
        Self::from_slice_with_dtype(values, shape, Some(ffi::DTYPE_I32))
    }
}

impl<T> DeviceTensor<T>
where
    T: Copy,
{
    fn from_slice_with_dtype(
        values: &[T],
        shape: Vec<usize>,
        dtype: Option<ffi::DTypeRaw>,
    ) -> VectorResult<Self> {
        let len = shape.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| VectorError::new("device tensor element count overflows"))
        })?;
        if values.len() != len {
            return Err(VectorError::new(format!(
                "device upload length mismatch: slice has {}, shape has {len}",
                values.len()
            )));
        }

        let mut ptr = ptr::null_mut();
        let bytes = mem::size_of_val(values);
        assert_cuda(
            unsafe { cudaMalloc(&mut ptr, bytes) },
            "allocate vector device tensor",
        );
        assert_cuda(
            unsafe {
                cudaMemcpy(
                    ptr,
                    values.as_ptr().cast(),
                    bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "upload vector device tensor",
        );
        Ok(Self {
            ptr: ptr.cast(),
            len,
            shape,
            dtype,
        })
    }
}

impl<T> DeviceTensor<T> {
    fn as_device_ptr(&self) -> ffi::DevicePtr {
        self.ptr.cast()
    }

    fn dtype(&self) -> VectorResult<ffi::DTypeRaw> {
        self.dtype
            .ok_or_else(|| VectorError::new("raw tensor has no qsfi dtype descriptor"))
    }

    fn tensor1(&self) -> VectorResult<ffi::Tensor1> {
        let shape = self.shape_array::<1>()?;
        let stride = contiguous_strides(shape)?;
        Ok(ffi::Tensor1 {
            data: self.as_device_ptr(),
            dtype: self.dtype()?,
            shape,
            stride,
        })
    }

    fn tensor2(&self) -> VectorResult<ffi::Tensor2> {
        let shape = self.shape_array::<2>()?;
        let stride = contiguous_strides(shape)?;
        Ok(ffi::Tensor2 {
            data: self.as_device_ptr(),
            dtype: self.dtype()?,
            shape,
            stride,
        })
    }

    fn tensor3(&self) -> VectorResult<ffi::Tensor3> {
        let shape = self.shape_array::<3>()?;
        let stride = contiguous_strides(shape)?;
        Ok(ffi::Tensor3 {
            data: self.as_device_ptr(),
            dtype: self.dtype()?,
            shape,
            stride,
        })
    }

    fn tensor4(&self) -> VectorResult<ffi::Tensor4> {
        let shape = self.shape_array::<4>()?;
        let stride = contiguous_strides(shape)?;
        Ok(ffi::Tensor4 {
            data: self.as_device_ptr(),
            dtype: self.dtype()?,
            shape,
            stride,
        })
    }

    fn shape_array<const N: usize>(&self) -> VectorResult<[i64; N]> {
        if self.shape.len() != N {
            return Err(VectorError::new(format!(
                "rank mismatch: tensor has rank {}, descriptor needs rank {N}",
                self.shape.len()
            )));
        }
        let mut out = [0_i64; N];
        for (idx, dim) in self.shape.iter().copied().enumerate() {
            out[idx] = i64::try_from(dim)
                .map_err(|_| VectorError::new("tensor dimension does not fit i64"))?;
        }
        Ok(out)
    }
}

impl<T> Drop for DeviceTensor<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr.cast());
            }
        }
    }
}

fn contiguous_strides<const N: usize>(shape: [i64; N]) -> VectorResult<[i64; N]> {
    let mut stride = [1_i64; N];
    let mut running = 1_i64;
    for idx in (0..N).rev() {
        stride[idx] = running;
        running = running
            .checked_mul(shape[idx])
            .ok_or_else(|| VectorError::new("contiguous stride overflows"))?;
    }
    Ok(stride)
}

#[repr(C)]
struct QsfiContextDesc {
    device_ordinal: i32,
    stream: *mut c_void,
}

struct QsfiContext {
    raw: *mut c_void,
}

impl QsfiContext {
    fn new() -> Self {
        assert_cuda(unsafe { cudaSetDevice(0) }, "set CUDA vector test device");
        let desc = QsfiContextDesc {
            device_ordinal: 0,
            stream: ptr::null_mut(),
        };
        let mut raw = ptr::null_mut();
        assert_qsfi_ok(
            unsafe { qsfi_context_create(&desc, &mut raw) },
            "create qsfi vector test context",
        );
        assert!(
            !raw.is_null(),
            "create qsfi vector test context returned null"
        );
        Self { raw }
    }

    fn raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for QsfiContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                qsfi_context_destroy(self.raw);
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

fn qsfi_status_name(status: ffi::StatusRaw) -> String {
    let ptr = unsafe { qsfi_status_string(status) };
    if ptr.is_null() {
        return format!("qsfi status {status}");
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn assert_qsfi_ok(status: ffi::StatusRaw, what: &str) {
    assert_eq!(
        status,
        QSFI_STATUS_OK,
        "{what}: {} ({status})",
        qsfi_status_name(status)
    );
}

fn set_cuda_test_device() {
    assert_cuda(unsafe { cudaSetDevice(0) }, "set CUDA vector test device");
}

fn absent_tensor1(dtype: ffi::DTypeRaw) -> ffi::Tensor1 {
    ffi::Tensor1 {
        data: ptr::null_mut(),
        dtype,
        shape: [0],
        stride: [0],
    }
}

fn absent_tensor2(dtype: ffi::DTypeRaw) -> ffi::Tensor2 {
    ffi::Tensor2 {
        data: ptr::null_mut(),
        dtype,
        shape: [0, 0],
        stride: [0, 0],
    }
}

fn parse_manifest(value: &JsonValue) -> VectorResult<VectorManifest> {
    let root = value.object("manifest root")?;
    let schema_version = required_i64(root, "schema_version")?;
    if schema_version != 1 {
        return Err(VectorError::new(format!(
            "unsupported schema_version {schema_version}"
        )));
    }

    let byte_order = required_string(root, "byte_order")?;
    if byte_order != "little" {
        return Err(VectorError::new(format!(
            "unsupported byte_order {byte_order:?}"
        )));
    }

    let name = required_string(root, "name")?.to_owned();
    let groups = required_string_array(root, "groups")?;
    let tensor_values = required_array(root, "tensors")?;
    let mut seen_names = HashSet::new();
    let mut seen_files = HashSet::new();
    let mut tensors = Vec::with_capacity(tensor_values.len());
    for (idx, tensor_value) in tensor_values.iter().enumerate() {
        let tensor = parse_tensor_spec(tensor_value, idx)?;
        if !seen_names.insert(tensor.name.clone()) {
            return Err(VectorError::new(format!(
                "duplicate tensor name {:?}",
                tensor.name
            )));
        }
        if !seen_files.insert(tensor.file.clone()) {
            return Err(VectorError::new(format!(
                "duplicate tensor file {:?}",
                tensor.file
            )));
        }
        tensors.push(tensor);
    }

    Ok(VectorManifest {
        name,
        groups,
        tensors,
    })
}

fn parse_tensor_spec(value: &JsonValue, idx: usize) -> VectorResult<TensorSpec> {
    let what = format!("tensor[{idx}]");
    let tensor = value.object(&what)?;
    let name = required_string(tensor, "name")?.to_owned();
    let dtype = VectorDType::parse(required_string(tensor, "dtype")?)?;
    let file = required_string(tensor, "file")?.to_owned();
    validate_flat_file_name(&file)?;
    let role = optional_string(tensor, "role")?.map(str::to_owned);
    let shape_values = required_array(tensor, "shape")?;
    let mut shape = Vec::with_capacity(shape_values.len());
    for (dim_idx, dim_value) in shape_values.iter().enumerate() {
        let dim = json_i64(dim_value, &format!("tensor {name} shape[{dim_idx}]"))?;
        if dim < 0 {
            return Err(VectorError::new(format!(
                "tensor {name} has negative shape dimension {dim}"
            )));
        }
        shape.push(
            usize::try_from(dim)
                .map_err(|_| VectorError::new(format!("tensor {name} dimension overflows")))?,
        );
    }

    Ok(TensorSpec {
        name,
        dtype,
        shape,
        file,
        role,
    })
}

fn validate_flat_file_name(file: &str) -> VectorResult<()> {
    let path = Path::new(file);
    if path.is_absolute() {
        return Err(VectorError::new(format!(
            "tensor file {file:?} must be relative"
        )));
    }
    if file.contains('/') || file.contains('\\') {
        return Err(VectorError::new(format!(
            "tensor file {file:?} must be flat, not nested"
        )));
    }
    if path.components().any(|component| {
        !matches!(
            component,
            path::Component::Normal(_) | path::Component::CurDir
        )
    }) {
        return Err(VectorError::new(format!(
            "tensor file {file:?} contains invalid path components"
        )));
    }
    Ok(())
}

fn required_array<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> VectorResult<&'a [JsonValue]> {
    object
        .get(key)
        .ok_or_else(|| VectorError::new(format!("missing required key {key:?}")))?
        .array(key)
}

fn required_i64(object: &BTreeMap<String, JsonValue>, key: &str) -> VectorResult<i64> {
    let value = object
        .get(key)
        .ok_or_else(|| VectorError::new(format!("missing required key {key:?}")))?;
    json_i64(value, key)
}

fn required_bool(object: &BTreeMap<String, JsonValue>, key: &str) -> VectorResult<bool> {
    object
        .get(key)
        .ok_or_else(|| VectorError::new(format!("missing required key {key:?}")))?
        .bool(key)
}

fn required_f32(object: &BTreeMap<String, JsonValue>, key: &str) -> VectorResult<f32> {
    let value = object
        .get(key)
        .ok_or_else(|| VectorError::new(format!("missing required key {key:?}")))?;
    json_f32(value, key)
}

fn required_object<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> VectorResult<&'a BTreeMap<String, JsonValue>> {
    object
        .get(key)
        .ok_or_else(|| VectorError::new(format!("missing required key {key:?}")))?
        .object(key)
}

fn required_string<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> VectorResult<&'a str> {
    object
        .get(key)
        .ok_or_else(|| VectorError::new(format!("missing required key {key:?}")))?
        .string(key)
}

fn optional_string<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> VectorResult<Option<&'a str>> {
    object.get(key).map(|value| value.string(key)).transpose()
}

fn required_string_array(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> VectorResult<Vec<String>> {
    required_array(object, key)?
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            value
                .string(&format!("{key}[{idx}]"))
                .map(ToOwned::to_owned)
        })
        .collect()
}

fn json_i64(value: &JsonValue, what: &str) -> VectorResult<i64> {
    let number = value.number(what)?;
    if number.contains('.') || number.contains('e') || number.contains('E') {
        return Err(VectorError::new(format!("{what} must be an integer")));
    }
    number
        .parse()
        .map_err(|_| VectorError::new(format!("{what} does not fit i64")))
}

fn json_f32(value: &JsonValue, what: &str) -> VectorResult<f32> {
    value
        .number(what)?
        .parse()
        .map_err(|_| VectorError::new(format!("{what} does not fit f32")))
}

fn find_manifest_paths(root: &Path) -> VectorResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    collect_manifest_paths(root, &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn collect_manifest_paths(dir: &Path, out: &mut Vec<PathBuf>) -> VectorResult<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_manifest_paths(&path, out)?;
        } else if file_type.is_file()
            && path.file_name().and_then(|name| name.to_str()) == Some("manifest.json")
        {
            out.push(path);
        }
    }
    Ok(())
}

fn vector_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_VECTOR_ROOT)
}

fn required_vector_root() -> PathBuf {
    let root = vector_root();
    assert!(
        root.is_dir(),
        "required Qwen3.6 correctness vector root is missing: {}. \
         Generate it with `just generate-vectors`.",
        root.display()
    );
    root
}

fn require_manifest(manifest: &Path) {
    assert!(
        manifest.is_file(),
        "required Qwen3.6 correctness vector manifest is missing: {}. \
         Generate vectors with `just generate-vectors`.",
        manifest.display()
    );
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn compute_standard_rmsnorm_f32(
    x_bf16: &[u16],
    raw_weight_bf16: &[u16],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(x_bf16.len(), rows * hidden);
    assert_eq!(raw_weight_bf16.len(), hidden);

    let raw_weight: Vec<f32> = raw_weight_bf16
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect();
    let x: Vec<f32> = x_bf16.iter().copied().map(bf16_bits_to_f32).collect();

    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(hidden) {
        let mean_sq = row.iter().map(|value| value * value).sum::<f32>() / hidden as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for (value, weight) in row.iter().zip(&raw_weight) {
            out.push(value * inv_rms * weight);
        }
    }
    out
}

fn load_json(path: &Path) -> VectorResult<JsonValue> {
    let text = fs::read_to_string(path)
        .map_err(|err| VectorError::new(format!("read {}: {err}", path.display())))?;
    JsonParser::new(&text).parse()
}

fn load_norm_mismatch_metadata(
    index_path: &Path,
    manifest_rel: &str,
) -> VectorResult<NormMismatchMetadata> {
    let root = load_json(index_path)?;
    let root_object = root.object("norm index root")?;
    let cases = required_array(root_object, "cases")?;
    for (idx, case) in cases.iter().enumerate() {
        let case_object = case.object(&format!("cases[{idx}]"))?;
        if required_string(case_object, "manifest")? != manifest_rel {
            continue;
        }
        let mismatch_count = required_i64(case_object, "standard_rmsnorm_bf16_mismatch_count")?;
        return Ok(NormMismatchMetadata {
            standard_rmsnorm_expected_fail: required_bool(
                case_object,
                "standard_rmsnorm_expected_fail",
            )?,
            standard_rmsnorm_bf16_mismatch_count: usize::try_from(mismatch_count)
                .map_err(|_| VectorError::new("mismatch count does not fit usize"))?,
        });
    }
    Err(VectorError::new(format!(
        "norm index {} is missing manifest {manifest_rel:?}",
        index_path.display()
    )))
}

fn load_manifest_eps(manifest_path: &Path) -> VectorResult<f32> {
    let root = load_json(manifest_path)?;
    let root_object = root.object("manifest root")?;
    let metadata = required_object(root_object, "metadata")?;
    required_f32(metadata, "eps")
}

#[derive(Debug)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    fn object(&self, what: &str) -> VectorResult<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(value) => Ok(value),
            _ => Err(VectorError::new(format!("{what} must be a JSON object"))),
        }
    }

    fn array(&self, what: &str) -> VectorResult<&[JsonValue]> {
        match self {
            Self::Array(value) => Ok(value),
            _ => Err(VectorError::new(format!("{what} must be a JSON array"))),
        }
    }

    fn string(&self, what: &str) -> VectorResult<&str> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(VectorError::new(format!("{what} must be a JSON string"))),
        }
    }

    fn bool(&self, what: &str) -> VectorResult<bool> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(VectorError::new(format!("{what} must be a JSON bool"))),
        }
    }

    fn number(&self, what: &str) -> VectorResult<&str> {
        match self {
            Self::Number(value) => Ok(value),
            _ => Err(VectorError::new(format!("{what} must be a JSON number"))),
        }
    }
}

struct JsonParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn parse(mut self) -> VectorResult<JsonValue> {
        let value = self.parse_value()?;
        self.skip_ws();
        if self.pos != self.input.len() {
            return Err(self.error("trailing characters"));
        }
        Ok(value)
    }

    fn parse_value(&mut self) -> VectorResult<JsonValue> {
        self.skip_ws();
        match self.peek_byte() {
            Some(b'n') => self.parse_literal(b"null", JsonValue::Null),
            Some(b't') => self.parse_literal(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_literal(b"false", JsonValue::Bool(false)),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-' | b'0'..=b'9') => self.parse_number().map(JsonValue::Number),
            Some(_) => Err(self.error("unexpected JSON token")),
            None => Err(self.error("unexpected end of JSON")),
        }
    }

    fn parse_literal(&mut self, literal: &[u8], value: JsonValue) -> VectorResult<JsonValue> {
        if self.input.as_bytes()[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            Ok(value)
        } else {
            Err(self.error("invalid JSON literal"))
        }
    }

    fn parse_array(&mut self) -> VectorResult<JsonValue> {
        self.expect_byte(b'[')?;
        self.skip_ws();
        let mut values = Vec::new();
        if self.consume_byte(b']') {
            return Ok(JsonValue::Array(values));
        }
        loop {
            values.push(self.parse_value()?);
            self.skip_ws();
            if self.consume_byte(b']') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(JsonValue::Array(values))
    }

    fn parse_object(&mut self) -> VectorResult<JsonValue> {
        self.expect_byte(b'{')?;
        self.skip_ws();
        let mut values = BTreeMap::new();
        if self.consume_byte(b'}') {
            return Ok(JsonValue::Object(values));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect_byte(b':')?;
            let value = self.parse_value()?;
            if values.insert(key.clone(), value).is_some() {
                return Err(self.error(&format!("duplicate JSON key {key:?}")));
            }
            self.skip_ws();
            if self.consume_byte(b'}') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(JsonValue::Object(values))
    }

    fn parse_string(&mut self) -> VectorResult<String> {
        self.expect_byte(b'"')?;
        let mut out = String::new();
        while let Some(byte) = self.next_byte() {
            match byte {
                b'"' => return Ok(out),
                b'\\' => out.push(self.parse_escape()?),
                0x00..=0x1f => return Err(self.error("control byte in JSON string")),
                _ => {
                    let start = self.pos - 1;
                    let str_len = self.input[start..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.error("invalid UTF-8 in JSON string"))?
                        .len_utf8();
                    out.push_str(&self.input[start..start + str_len]);
                    self.pos = start + str_len;
                }
            }
        }
        Err(self.error("unterminated JSON string"))
    }

    fn parse_escape(&mut self) -> VectorResult<char> {
        let byte = self
            .next_byte()
            .ok_or_else(|| self.error("unterminated JSON escape"))?;
        match byte {
            b'"' => Ok('"'),
            b'\\' => Ok('\\'),
            b'/' => Ok('/'),
            b'b' => Ok('\u{0008}'),
            b'f' => Ok('\u{000c}'),
            b'n' => Ok('\n'),
            b'r' => Ok('\r'),
            b't' => Ok('\t'),
            b'u' => self.parse_unicode_escape(),
            _ => Err(self.error("invalid JSON escape")),
        }
    }

    fn parse_unicode_escape(&mut self) -> VectorResult<char> {
        let code = self.parse_hex4()?;
        if (0xD800..=0xDBFF).contains(&code) {
            let save = self.pos;
            if self.consume_byte(b'\\') && self.consume_byte(b'u') {
                let low = self.parse_hex4()?;
                if (0xDC00..=0xDFFF).contains(&low) {
                    let scalar = 0x1_0000 + (((code - 0xD800) << 10) | (low - 0xDC00));
                    return char::from_u32(scalar)
                        .ok_or_else(|| self.error("invalid unicode scalar"));
                }
            }
            self.pos = save;
            return Err(self.error("invalid unicode surrogate pair"));
        }
        char::from_u32(code).ok_or_else(|| self.error("invalid unicode scalar"))
    }

    fn parse_hex4(&mut self) -> VectorResult<u32> {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = self
                .next_byte()
                .ok_or_else(|| self.error("unterminated unicode escape"))?;
            value = (value << 4)
                | match byte {
                    b'0'..=b'9' => u32::from(byte - b'0'),
                    b'a'..=b'f' => u32::from(byte - b'a') + 10,
                    b'A'..=b'F' => u32::from(byte - b'A') + 10,
                    _ => return Err(self.error("invalid unicode escape digit")),
                };
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> VectorResult<String> {
        let start = self.pos;
        self.consume_byte(b'-');
        match self.peek_byte() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.error("invalid JSON number")),
        }
        if self.consume_byte(b'.') {
            if !matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid JSON number fraction"));
            }
            while matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek_byte(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek_byte(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid JSON number exponent"));
            }
            while matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        Ok(self.input[start..self.pos].to_owned())
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek_byte(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn expect_byte(&mut self, expected: u8) -> VectorResult<()> {
        if self.consume_byte(expected) {
            Ok(())
        } else {
            Err(self.error(&format!("expected byte {:?}", char::from(expected))))
        }
    }

    fn consume_byte(&mut self, expected: u8) -> bool {
        if self.peek_byte() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }

    fn next_byte(&mut self) -> Option<u8> {
        let byte = self.peek_byte()?;
        self.pos += 1;
        Some(byte)
    }

    fn error(&self, message: &str) -> VectorError {
        VectorError::new(format!("JSON parse error at byte {}: {message}", self.pos))
    }
}

#[test]
fn qwen36_vector_manifests_load_flat_tensors() {
    let root = required_vector_root();
    let manifests = find_manifest_paths(&root).unwrap();
    assert!(
        !manifests.is_empty(),
        "no vector manifests found under {}",
        root.display()
    );

    for manifest_path in manifests {
        let artifact = VectorArtifact::load(&manifest_path).unwrap();
        assert_eq!(artifact.manifest.tensors.len(), artifact.tensors.len());
        for spec in &artifact.manifest.tensors {
            let tensor = artifact.tensor(&spec.name).unwrap();
            assert_eq!(tensor.bytes.len(), spec.byte_len().unwrap());
        }
    }
}

#[test]
fn norm_debug_vector_exposes_typed_views() {
    let root = required_vector_root();
    let manifest = root.join("norms/gemma_rmsnorm_dim8_debug/manifest.json");
    require_manifest(&manifest);

    let artifact = VectorArtifact::load(manifest).unwrap();
    let x = artifact.tensor("x").unwrap();
    let raw_weight = artifact.tensor("raw_weight").unwrap();
    let expected_bf16 = artifact.tensor("expected_output_bf16").unwrap();
    let expected_f32 = artifact.tensor("expected_output_f32").unwrap();

    assert_eq!(x.spec.shape, [2, 8]);
    assert_eq!(x.as_bf16_words().unwrap().len(), 16);
    assert_eq!(raw_weight.as_bf16_words().unwrap().len(), 8);
    assert_eq!(expected_bf16.as_bf16_words().unwrap().len(), 16);
    assert_eq!(expected_f32.as_f32_vec().unwrap().len(), 16);
}

#[test]
fn qwen36_norm_vectors_fail_standard_rmsnorm() {
    let root = required_vector_root();
    let manifest_rel = "gemma_rmsnorm_dim8_debug/manifest.json";
    let manifest = root.join("norms").join(manifest_rel);
    require_manifest(&manifest);

    let artifact = VectorArtifact::load(&manifest).unwrap();
    let mismatch =
        load_norm_mismatch_metadata(&root.join("norms/index.json"), manifest_rel).unwrap();
    assert!(mismatch.standard_rmsnorm_expected_fail);

    let x = artifact.tensor("x").unwrap();
    let raw_weight = artifact.tensor("raw_weight").unwrap();
    let expected_output_bf16 = artifact.tensor("expected_output_bf16").unwrap();
    let standard_output_bf16 = artifact.tensor("standard_output_bf16").unwrap();

    assert_eq!(x.spec.shape, [2, 8]);
    assert_eq!(raw_weight.spec.shape, [8]);

    let eps = load_manifest_eps(&manifest).unwrap();
    let computed_standard_f32 = compute_standard_rmsnorm_f32(
        &x.as_bf16_words().unwrap(),
        &raw_weight.as_bf16_words().unwrap(),
        x.spec.shape[0],
        x.spec.shape[1],
        eps,
    );
    let computed_standard_bf16: Vec<u16> = computed_standard_f32
        .into_iter()
        .map(f32_to_bf16_bits)
        .collect();

    assert_eq!(
        computed_standard_bf16,
        standard_output_bf16.as_bf16_words().unwrap()
    );

    let expected_output_bf16 = expected_output_bf16.as_bf16_words().unwrap();
    let mismatch_count = computed_standard_bf16
        .iter()
        .zip(expected_output_bf16.iter())
        .filter(|(computed, expected)| computed != expected)
        .count();
    assert_eq!(
        mismatch_count,
        mismatch.standard_rmsnorm_bf16_mismatch_count
    );
    assert!(mismatch_count > 0);
}

fn load_norm_vector_case(case: &str) -> Option<(PathBuf, VectorArtifact)> {
    let root = required_vector_root();
    let manifest = root.join("norms").join(case).join("manifest.json");
    require_manifest(&manifest);
    let artifact = VectorArtifact::load(&manifest).unwrap();
    Some((manifest, artifact))
}

fn load_gdn_post_conv_prep_vector() -> Option<(PathBuf, VectorArtifact)> {
    let root = required_vector_root();
    let manifest = root.join("gdn_post_conv_prep/manifest.json");
    require_manifest(&manifest);
    let artifact = VectorArtifact::load(&manifest).unwrap();
    Some((manifest, artifact))
}

fn shape2(tensor: &RawTensor, name: &str) -> (usize, usize) {
    assert_eq!(tensor.spec.shape.len(), 2, "{name} must be a rank-2 tensor");
    (tensor.spec.shape[0], tensor.spec.shape[1])
}

fn shape3(tensor: &RawTensor, name: &str) -> (usize, usize, usize) {
    assert_eq!(tensor.spec.shape.len(), 3, "{name} must be a rank-3 tensor");
    (
        tensor.spec.shape[0],
        tensor.spec.shape[1],
        tensor.spec.shape[2],
    )
}

fn shape4(tensor: &RawTensor, name: &str) -> (usize, usize, usize, usize) {
    assert_eq!(tensor.spec.shape.len(), 4, "{name} must be a rank-4 tensor");
    (
        tensor.spec.shape[0],
        tensor.spec.shape[1],
        tensor.spec.shape[2],
        tensor.spec.shape[3],
    )
}

fn gather_compact_state_slots(pool: &[u16], slots: &[i32], state_size: usize) -> Vec<u16> {
    let mut compact = Vec::with_capacity(slots.len() * state_size);
    for &slot in slots {
        let slot = usize::try_from(slot).expect("state slot must be non-negative");
        let begin = slot
            .checked_mul(state_size)
            .expect("state slot offset overflow");
        let end = begin + state_size;
        compact.extend_from_slice(&pool[begin..end]);
    }
    compact
}

fn case_ping_pong_slots(case_count: usize, slot: usize) -> Vec<i32> {
    (0..case_count)
        .map(|case| i32::try_from(case * 2 + slot).unwrap())
        .collect()
}

fn assert_u16_words_eq(what: &str, got: &[u16], expected: &[u16]) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: got len {}, expected len {}",
        got.len(),
        expected.len()
    );
    let mismatch_count = got
        .iter()
        .zip(expected)
        .filter(|(got, expected)| got != expected)
        .count();
    if mismatch_count == 0 {
        return;
    }
    let first = got
        .iter()
        .zip(expected)
        .position(|(got, expected)| got != expected)
        .unwrap();
    panic!(
        "{what}: {mismatch_count} mismatches; first at {first}: got 0x{:04x}, expected 0x{:04x}",
        got[first], expected[first]
    );
}

fn assert_f32_close(what: &str, got: &[f32], expected: &[f32], abs_tol: f32) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: got len {}, expected len {}",
        got.len(),
        expected.len()
    );
    let mut mismatch_count = 0usize;
    let mut first_mismatch = None;
    let mut max_abs_delta = 0.0f32;
    let mut max_abs_delta_idx = 0usize;
    for (idx, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let delta = (got - expected).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            max_abs_delta_idx = idx;
        }
        if delta > abs_tol {
            mismatch_count += 1;
            first_mismatch.get_or_insert((idx, got, expected, delta));
        }
    }
    if let Some((idx, got, expected, delta)) = first_mismatch {
        panic!(
            "{what}: {mismatch_count} mismatches; first at {idx}: got {got:.8}, \
             expected {expected:.8}, delta {delta:.8}, abs_tol {abs_tol:.8}; \
             max_abs_delta {max_abs_delta:.8} at {max_abs_delta_idx}"
        );
    }
}

fn assert_bf16_close_to_f32_oracle(
    what: &str,
    got_bf16: &[u16],
    expected_bf16: &[u16],
    expected_f32: &[f32],
    abs_tol: f32,
) {
    assert_eq!(
        got_bf16.len(),
        expected_f32.len(),
        "{what}: got len {}, expected f32 len {}",
        got_bf16.len(),
        expected_f32.len()
    );
    assert_eq!(
        got_bf16.len(),
        expected_bf16.len(),
        "{what}: got len {}, expected bf16 len {}",
        got_bf16.len(),
        expected_bf16.len()
    );

    let bf16_mismatch_count = got_bf16
        .iter()
        .zip(expected_bf16)
        .filter(|(got, expected)| got != expected)
        .count();
    let mut max_abs_delta = 0.0f32;
    let mut max_abs_delta_idx = 0usize;
    for (idx, (&got, &expected)) in got_bf16.iter().zip(expected_f32).enumerate() {
        let delta = (bf16_bits_to_f32(got) - expected).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            max_abs_delta_idx = idx;
        }
    }

    for (idx, ((&got, &expected_bits), &expected)) in got_bf16
        .iter()
        .zip(expected_bf16)
        .zip(expected_f32)
        .enumerate()
    {
        let got_f32 = bf16_bits_to_f32(got);
        let delta = (got_f32 - expected).abs();
        assert!(
            got == expected_bits || delta <= abs_tol,
            "{what}[{idx}]: got {got_f32:.8} (0x{got:04x}), expected {expected:.8} \
             +/- {abs_tol:.8} or expected_bf16=0x{expected_bits:04x}; \
             bf16_mismatch_count={bf16_mismatch_count}, \
             max_abs_delta={max_abs_delta:.8} at {max_abs_delta_idx}"
        );
    }
}

#[test]
fn qwen36_cuda_rmsnorm_hidden2048_matches_gemma_vector() {
    let Some((manifest, artifact)) = load_norm_vector_case("gemma_rmsnorm_hidden2048") else {
        return;
    };

    let x_raw = artifact.tensor("x").unwrap();
    let raw_weight_raw = artifact.tensor("raw_weight").unwrap();
    let expected_output_bf16 = artifact
        .tensor("expected_output_bf16")
        .unwrap()
        .as_bf16_words()
        .unwrap();
    let expected_output_f32 = artifact
        .tensor("expected_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let (rows, hidden) = shape2(x_raw, "x");
    assert_eq!(hidden, 2048);
    assert_eq!(raw_weight_raw.spec.shape, [hidden]);

    let x = DeviceTensor::from_bf16(x_raw).unwrap();
    let raw_weight = DeviceTensor::from_bf16(raw_weight_raw).unwrap();
    let out = DeviceTensor::zeroed_bf16(x_raw.spec.shape.clone()).unwrap();
    let ctx = QsfiContext::new();
    let eps = load_manifest_eps(&manifest).unwrap();
    let desc = ffi::RmsnormDesc {
        x: x.tensor2().unwrap(),
        weight: raw_weight.tensor1().unwrap(),
        out: out.tensor2().unwrap(),
        hidden_size: u32::try_from(hidden).unwrap(),
        weight_bias: 1.0,
        eps,
    };

    assert_qsfi_ok(
        unsafe { qsfi_rmsnorm(ctx.raw(), &desc) },
        "launch qwen36 gemma RMSNorm vector through ordinary qsfi rmsnorm",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 gemma RMSNorm vector",
    );

    let got = out.to_u16_vec("download qwen36 gemma RMSNorm output");
    assert_eq!(got.len(), rows * hidden);
    assert_bf16_close_to_f32_oracle(
        "gemma_rmsnorm_hidden2048 output",
        &got,
        &expected_output_bf16,
        &expected_output_f32,
        BF16_NORM_ABS_TOL,
    );
}

#[test]
fn qwen36_cuda_fused_add_rmsnorm_hidden2048_matches_gemma_vector() {
    let Some((manifest, artifact)) = load_norm_vector_case("gemma_fused_add_rmsnorm_hidden2048")
    else {
        return;
    };

    let x_raw = artifact.tensor("x").unwrap();
    let residual_raw = artifact.tensor("residual").unwrap();
    let raw_weight_raw = artifact.tensor("raw_weight").unwrap();
    let expected_output_bf16 = artifact
        .tensor("expected_output_bf16")
        .unwrap()
        .as_bf16_words()
        .unwrap();
    let expected_output_f32 = artifact
        .tensor("expected_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_residual_bf16 = artifact
        .tensor("expected_residual_out_bf16")
        .unwrap()
        .as_bf16_words()
        .unwrap();
    let expected_residual_f32 = artifact
        .tensor("expected_residual_out_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let (rows, hidden) = shape2(x_raw, "x");
    assert_eq!(hidden, 2048);
    assert_eq!(residual_raw.spec.shape, x_raw.spec.shape);
    assert_eq!(raw_weight_raw.spec.shape, [hidden]);

    let x = DeviceTensor::from_bf16(x_raw).unwrap();
    let residual = DeviceTensor::from_bf16(residual_raw).unwrap();
    let raw_weight = DeviceTensor::from_bf16(raw_weight_raw).unwrap();
    let ctx = QsfiContext::new();
    let eps = load_manifest_eps(&manifest).unwrap();
    let desc = ffi::FusedAddRmsnormDesc {
        x: x.tensor2().unwrap(),
        residual_inout: residual.tensor2().unwrap(),
        weight: raw_weight.tensor1().unwrap(),
        out: x.tensor2().unwrap(),
        hidden_size: u32::try_from(hidden).unwrap(),
        weight_bias: 1.0,
        eps,
    };

    assert_qsfi_ok(
        unsafe { qsfi_fused_add_rmsnorm(ctx.raw(), &desc) },
        "launch qwen36 gemma fused-add RMSNorm vector through ordinary qsfi fused-add rmsnorm",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 gemma fused-add RMSNorm vector",
    );

    let got_residual = residual.to_u16_vec("download qwen36 gemma fused-add residual");
    assert_eq!(got_residual.len(), rows * hidden);
    assert_bf16_close_to_f32_oracle(
        "gemma_fused_add_rmsnorm_hidden2048 residual",
        &got_residual,
        &expected_residual_bf16,
        &expected_residual_f32,
        BF16_RESIDUAL_ABS_TOL,
    );

    let got_output = x.to_u16_vec("download qwen36 gemma fused-add RMSNorm output");
    assert_eq!(got_output.len(), rows * hidden);
    assert_bf16_close_to_f32_oracle(
        "gemma_fused_add_rmsnorm_hidden2048 output",
        &got_output,
        &expected_output_bf16,
        &expected_output_f32,
        BF16_NORM_ABS_TOL,
    );
}

#[test]
fn qwen36_cuda_gdn_causal_conv_width4_matches_vector() {
    let Some((_manifest, artifact)) = load_gdn_post_conv_prep_vector() else {
        return;
    };

    let x_raw = artifact.tensor("mixed_qkv").unwrap();
    let weight_raw = artifact.tensor("conv_weight").unwrap();
    let case_lengths_raw = artifact.tensor("case_lengths").unwrap();
    let case_offsets_raw = artifact.tensor("case_offsets").unwrap();
    let expected_out_raw = artifact.tensor("conv_output_bf16").unwrap();
    let expected_out_bf16 = expected_out_raw.as_bf16_words().unwrap();
    let expected_out_f32 = artifact
        .tensor("conv_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_state_raw = artifact.tensor("conv_final_state").unwrap();
    let expected_state = expected_state_raw.as_bf16_words().unwrap();

    let (tokens, packed_dim) = shape2(x_raw, "mixed_qkv");
    assert_eq!(packed_dim, 8192);
    assert_eq!(weight_raw.spec.shape, [packed_dim, 4]);
    assert_eq!(expected_out_raw.spec.shape, x_raw.spec.shape);
    let (state_pool, state_packed_dim, state_width) =
        shape3(expected_state_raw, "conv_final_state");
    assert_eq!(state_packed_dim, packed_dim);
    assert_eq!(state_width, 3);
    assert_eq!(case_lengths_raw.spec.shape, [state_pool]);
    assert_eq!(case_offsets_raw.spec.shape, [state_pool + 1]);

    set_cuda_test_device();
    let x = DeviceTensor::from_bf16(x_raw).unwrap();
    let weight = DeviceTensor::from_bf16(weight_raw).unwrap();
    let state = DeviceTensor::zeroed_bf16(expected_state_raw.spec.shape.clone()).unwrap();
    let read_indices =
        DeviceTensor::<i32>::from_i32_slice(&vec![-1_i32; state_pool], vec![state_pool]).unwrap();
    let write_indices: Vec<i32> = (0..state_pool)
        .map(|idx| i32::try_from(idx).unwrap())
        .collect();
    let write_indices =
        DeviceTensor::<i32>::from_i32_slice(&write_indices, vec![state_pool]).unwrap();
    let seq_indptr = DeviceTensor::from_i32(case_offsets_raw).unwrap();
    let out = DeviceTensor::zeroed_bf16(x_raw.spec.shape.clone()).unwrap();

    let desc = QscuQwen36GdnCausalConv1dDesc {
        x: x.tensor2().unwrap(),
        weight: weight.tensor2().unwrap(),
        bias: absent_tensor1(ffi::DTYPE_BF16),
        state: state.tensor3().unwrap(),
        state_read_indices: read_indices.tensor1().unwrap(),
        state_write_indices: write_indices.tensor1().unwrap(),
        seq_indptr: seq_indptr.as_device_ptr(),
        out: out.tensor2().unwrap(),
        num_tokens: u32::try_from(tokens).unwrap(),
        batch_size: u32::try_from(state_pool).unwrap(),
        activation: QSCU_ACTIVATION_SILU,
        update_state: 1,
    };

    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_causal_conv1d_bf16(&desc, ptr::null_mut()) },
        "launch qwen36 GDN causal conv vector",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 GDN causal conv vector",
    );

    let got_out = out.to_u16_vec("download qwen36 GDN causal conv output");
    assert_bf16_close_to_f32_oracle(
        "gdn causal conv output",
        &got_out,
        &expected_out_bf16,
        &expected_out_f32,
        BF16_GDN_CONV_ABS_TOL,
    );
    let got_state = state.to_u16_vec("download qwen36 GDN causal conv final state");
    assert_u16_words_eq("gdn causal conv final state", &got_state, &expected_state);
}

#[test]
fn qwen36_cuda_gdn_post_conv_raw_split_and_gates_match_vector() {
    let Some((_manifest, artifact)) = load_gdn_post_conv_prep_vector() else {
        return;
    };

    let conv_out_raw = artifact.tensor("conv_output_bf16").unwrap();
    let a_raw = artifact.tensor("a").unwrap();
    let b_raw = artifact.tensor("b").unwrap();
    let a_log_raw = artifact.tensor("A_log").unwrap();
    let dt_bias_raw = artifact.tensor("dt_bias").unwrap();
    let q_raw = artifact.tensor("q_raw").unwrap();
    let k_raw = artifact.tensor("k_raw").unwrap();
    let v_raw = artifact.tensor("v_raw").unwrap();
    let expected_q = q_raw.as_bf16_words().unwrap();
    let expected_k = k_raw.as_bf16_words().unwrap();
    let expected_v = v_raw.as_bf16_words().unwrap();
    let expected_g = artifact.tensor("g").unwrap().as_f32_vec().unwrap();
    let expected_decay = artifact.tensor("decay_exp").unwrap().as_f32_vec().unwrap();
    let expected_beta = artifact.tensor("beta").unwrap().as_f32_vec().unwrap();

    let (tokens, packed_dim) = shape2(conv_out_raw, "conv_output_bf16");
    assert_eq!(packed_dim, 8192);
    assert_eq!(a_raw.spec.shape, [tokens, 32]);
    assert_eq!(b_raw.spec.shape, [tokens, 32]);
    assert_eq!(a_log_raw.spec.shape, [32]);
    assert_eq!(dt_bias_raw.spec.shape, [32]);
    assert_eq!(shape3(q_raw, "q_raw"), (tokens, 16, 128));
    assert_eq!(shape3(k_raw, "k_raw"), (tokens, 16, 128));
    assert_eq!(shape3(v_raw, "v_raw"), (tokens, 32, 128));

    set_cuda_test_device();
    let conv_out = DeviceTensor::from_bf16(conv_out_raw).unwrap();
    let a = DeviceTensor::from_bf16(a_raw).unwrap();
    let b = DeviceTensor::from_bf16(b_raw).unwrap();
    let a_log = DeviceTensor::from_f32(a_log_raw).unwrap();
    let dt_bias = DeviceTensor::from_f32(dt_bias_raw).unwrap();
    let q = DeviceTensor::zeroed_bf16(q_raw.spec.shape.clone()).unwrap();
    let k = DeviceTensor::zeroed_bf16(k_raw.spec.shape.clone()).unwrap();
    let v = DeviceTensor::zeroed_bf16(v_raw.spec.shape.clone()).unwrap();
    let g = DeviceTensor::zeroed_f32(vec![tokens, 32]).unwrap();
    let beta = DeviceTensor::zeroed_f32(vec![tokens, 32]).unwrap();

    let mut desc = QscuQwen36GdnPostConvPrepareDesc {
        conv_out: conv_out.tensor2().unwrap(),
        a: a.tensor2().unwrap(),
        b: b.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        q: q.tensor3().unwrap(),
        k: k.tensor3().unwrap(),
        v: v.tensor3().unwrap(),
        g_out: g.tensor2().unwrap(),
        beta_out: beta.tensor2().unwrap(),
        num_tokens: u32::try_from(tokens).unwrap(),
        apply_qk_l2norm: 0,
        l2norm_eps: 1.0e-6,
        forget_gate_output: QSCU_GDN_FORGET_LOG_DECAY,
    };

    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_post_conv_prepare_bf16(&desc, ptr::null_mut()) },
        "launch qwen36 GDN post-conv raw split vector",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 GDN post-conv raw split vector",
    );

    let got_q = q.to_u16_vec("download qwen36 GDN raw q");
    let got_k = k.to_u16_vec("download qwen36 GDN raw k");
    let got_v = v.to_u16_vec("download qwen36 GDN raw v");
    let got_g = g.to_f32_vec("download qwen36 GDN log decay");
    let got_beta = beta.to_f32_vec("download qwen36 GDN beta");
    assert_u16_words_eq("gdn post-conv q_raw", &got_q, &expected_q);
    assert_u16_words_eq("gdn post-conv k_raw", &got_k, &expected_k);
    assert_u16_words_eq("gdn post-conv v_raw", &got_v, &expected_v);
    assert_f32_close("gdn post-conv g", &got_g, &expected_g, GDN_F32_ABS_TOL);
    assert_f32_close(
        "gdn post-conv beta",
        &got_beta,
        &expected_beta,
        GDN_F32_ABS_TOL,
    );

    desc.forget_gate_output = QSCU_GDN_FORGET_LINEAR_ALPHA;
    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_post_conv_prepare_bf16(&desc, ptr::null_mut()) },
        "launch qwen36 GDN post-conv decay-exp vector",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 GDN post-conv decay-exp vector",
    );
    let got_decay = g.to_f32_vec("download qwen36 GDN decay-exp");
    assert_f32_close(
        "gdn post-conv decay_exp",
        &got_decay,
        &expected_decay,
        GDN_F32_ABS_TOL,
    );
}

#[test]
fn qwen36_cuda_gdn_post_conv_l2norm_matches_vector() {
    let Some((_manifest, artifact)) = load_gdn_post_conv_prep_vector() else {
        return;
    };

    let conv_out_raw = artifact.tensor("conv_output_bf16").unwrap();
    let a_raw = artifact.tensor("a").unwrap();
    let b_raw = artifact.tensor("b").unwrap();
    let a_log_raw = artifact.tensor("A_log").unwrap();
    let dt_bias_raw = artifact.tensor("dt_bias").unwrap();
    let q_l2norm_raw = artifact.tensor("q_l2norm_bf16").unwrap();
    let k_l2norm_raw = artifact.tensor("k_l2norm_bf16").unwrap();
    let v_raw = artifact.tensor("v_raw").unwrap();
    let expected_q_bf16 = q_l2norm_raw.as_bf16_words().unwrap();
    let expected_q_f32 = artifact
        .tensor("q_l2norm_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_k_bf16 = k_l2norm_raw.as_bf16_words().unwrap();
    let expected_k_f32 = artifact
        .tensor("k_l2norm_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_v = v_raw.as_bf16_words().unwrap();

    let (tokens, packed_dim) = shape2(conv_out_raw, "conv_output_bf16");
    assert_eq!(packed_dim, 8192);
    assert_eq!(q_l2norm_raw.spec.shape, [tokens, 16, 128]);
    assert_eq!(k_l2norm_raw.spec.shape, [tokens, 16, 128]);
    assert_eq!(v_raw.spec.shape, [tokens, 32, 128]);

    set_cuda_test_device();
    let conv_out = DeviceTensor::from_bf16(conv_out_raw).unwrap();
    let a = DeviceTensor::from_bf16(a_raw).unwrap();
    let b = DeviceTensor::from_bf16(b_raw).unwrap();
    let a_log = DeviceTensor::from_f32(a_log_raw).unwrap();
    let dt_bias = DeviceTensor::from_f32(dt_bias_raw).unwrap();
    let q = DeviceTensor::zeroed_bf16(q_l2norm_raw.spec.shape.clone()).unwrap();
    let k = DeviceTensor::zeroed_bf16(k_l2norm_raw.spec.shape.clone()).unwrap();
    let v = DeviceTensor::zeroed_bf16(v_raw.spec.shape.clone()).unwrap();

    let desc = QscuQwen36GdnPostConvPrepareDesc {
        conv_out: conv_out.tensor2().unwrap(),
        a: a.tensor2().unwrap(),
        b: b.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        q: q.tensor3().unwrap(),
        k: k.tensor3().unwrap(),
        v: v.tensor3().unwrap(),
        g_out: absent_tensor2(ffi::DTYPE_F32),
        beta_out: absent_tensor2(ffi::DTYPE_F32),
        num_tokens: u32::try_from(tokens).unwrap(),
        apply_qk_l2norm: 1,
        l2norm_eps: 1.0e-6,
        forget_gate_output: QSCU_GDN_FORGET_LOG_DECAY,
    };

    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_post_conv_prepare_bf16(&desc, ptr::null_mut()) },
        "launch qwen36 GDN post-conv L2 norm vector",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 GDN post-conv L2 norm vector",
    );

    let got_q = q.to_u16_vec("download qwen36 GDN l2norm q");
    let got_k = k.to_u16_vec("download qwen36 GDN l2norm k");
    let got_v = v.to_u16_vec("download qwen36 GDN l2norm v");
    assert_bf16_close_to_f32_oracle(
        "gdn post-conv q_l2norm",
        &got_q,
        &expected_q_bf16,
        &expected_q_f32,
        BF16_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "gdn post-conv k_l2norm",
        &got_k,
        &expected_k_bf16,
        &expected_k_f32,
        BF16_NORM_ABS_TOL,
    );
    assert_u16_words_eq("gdn post-conv l2norm v_raw", &got_v, &expected_v);
}

#[test]
fn qwen36_cuda_gdn_gated_rmsnorm_silu_matches_vector() {
    let Some((_manifest, artifact)) = load_gdn_post_conv_prep_vector() else {
        return;
    };

    let x_raw = artifact.tensor("v_raw").unwrap();
    let gate_raw = artifact.tensor("z").unwrap();
    let weight_raw = artifact.tensor("gated_rmsnorm_weight").unwrap();
    let expected_out_raw = artifact.tensor("gated_rmsnorm_output_bf16").unwrap();
    let expected_out_bf16 = expected_out_raw.as_bf16_words().unwrap();
    let expected_out_f32 = artifact
        .tensor("gated_rmsnorm_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();

    let (tokens, num_v_heads, value_dim) = shape3(x_raw, "v_raw");
    assert_eq!((num_v_heads, value_dim), (32, 128));
    assert_eq!(gate_raw.spec.shape, x_raw.spec.shape);
    assert_eq!(weight_raw.spec.shape, [value_dim]);
    assert_eq!(expected_out_raw.spec.shape, x_raw.spec.shape);

    set_cuda_test_device();
    let x = DeviceTensor::from_bf16(x_raw).unwrap();
    let gate = DeviceTensor::from_bf16(gate_raw).unwrap();
    let weight = DeviceTensor::from_bf16(weight_raw).unwrap();
    let out = DeviceTensor::zeroed_bf16(x_raw.spec.shape.clone()).unwrap();

    let desc = QscuQwen36GdnRmsnormGatedDesc {
        x: x.tensor3().unwrap(),
        gate: gate.tensor3().unwrap(),
        weight: weight.tensor1().unwrap(),
        out: out.tensor3().unwrap(),
        num_tokens: u32::try_from(tokens).unwrap(),
        eps: 1.0e-6,
        gate_activation: QSCU_ACTIVATION_SILU,
    };

    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_rmsnorm_gated_bf16(&desc, ptr::null_mut()) },
        "launch qwen36 GDN gated RMSNorm vector",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 GDN gated RMSNorm vector",
    );

    let got_out = out.to_u16_vec("download qwen36 GDN gated RMSNorm output");
    assert_bf16_close_to_f32_oracle(
        "gdn gated rmsnorm output",
        &got_out,
        &expected_out_bf16,
        &expected_out_f32,
        BF16_GDN_NORM_ABS_TOL,
    );
}

#[test]
fn qscu_gdn_prefill_recurrence_matches_vector() {
    let Some((_manifest, artifact)) = load_gdn_post_conv_prep_vector() else {
        return;
    };

    let q_raw = artifact.tensor("q_raw").unwrap();
    let k_raw = artifact.tensor("k_raw").unwrap();
    let v_raw = artifact.tensor("v_raw").unwrap();
    let a_raw = artifact.tensor("a").unwrap();
    let b_raw = artifact.tensor("b").unwrap();
    let a_log_raw = artifact.tensor("A_log").unwrap();
    let dt_bias_raw = artifact.tensor("dt_bias").unwrap();
    let case_offsets_raw = artifact.tensor("case_offsets").unwrap();
    let expected_out_raw = artifact.tensor("recurrent_output_bf16").unwrap();
    let expected_out_bf16 = expected_out_raw.as_bf16_words().unwrap();
    let expected_out_f32 = artifact
        .tensor("recurrent_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_state_raw = artifact.tensor("recurrent_final_state_bf16").unwrap();
    let expected_state_bf16 = expected_state_raw.as_bf16_words().unwrap();
    let expected_state_f32 = artifact
        .tensor("recurrent_final_state_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();

    let (tokens, q_heads, key_dim) = shape3(q_raw, "q_raw");
    assert_eq!((q_heads, key_dim), (16, 128));
    assert_eq!(shape3(k_raw, "k_raw"), (tokens, 16, key_dim));
    assert_eq!(shape3(v_raw, "v_raw"), (tokens, 32, 128));
    assert_eq!(a_raw.spec.shape, [tokens, 32]);
    assert_eq!(b_raw.spec.shape, [tokens, 32]);
    assert_eq!(a_log_raw.spec.shape, [32]);
    assert_eq!(dt_bias_raw.spec.shape, [32]);
    assert_eq!(expected_out_raw.spec.shape, [tokens, 32, 128]);
    let (case_count, state_heads, state_value_dim, state_key_dim) =
        shape4(expected_state_raw, "recurrent_final_state_bf16");
    assert_eq!(
        (state_heads, state_value_dim, state_key_dim),
        (32, 128, 128)
    );
    assert_eq!(case_offsets_raw.spec.shape, [case_count + 1]);

    set_cuda_test_device();
    let q = DeviceTensor::from_bf16(q_raw).unwrap();
    let k = DeviceTensor::from_bf16(k_raw).unwrap();
    let v = DeviceTensor::from_bf16(v_raw).unwrap();
    let a = DeviceTensor::from_bf16(a_raw).unwrap();
    let b = DeviceTensor::from_bf16(b_raw).unwrap();
    let a_log = DeviceTensor::from_f32(a_log_raw).unwrap();
    let dt_bias = DeviceTensor::from_f32(dt_bias_raw).unwrap();
    let seq_indptr = DeviceTensor::from_i32(case_offsets_raw).unwrap();
    let state = DeviceTensor::zeroed_bf16(expected_state_raw.spec.shape.clone()).unwrap();
    let state_indices: Vec<i32> = (0..case_count)
        .map(|idx| i32::try_from(idx).unwrap())
        .collect();
    let state_indices =
        DeviceTensor::<i32>::from_i32_slice(&state_indices, vec![case_count]).unwrap();
    let out = DeviceTensor::zeroed_bf16(expected_out_raw.spec.shape.clone()).unwrap();
    let ctx = QsfiContext::new();

    let desc = QscuGdnPrefillDesc {
        q: q.tensor3().unwrap(),
        k: k.tensor3().unwrap(),
        v: v.tensor3().unwrap(),
        a: a.tensor2().unwrap(),
        b: b.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        state: state.tensor4().unwrap(),
        seq_indptr: seq_indptr.as_device_ptr(),
        state_indices: state_indices.tensor1().unwrap(),
        state_out_indices: absent_tensor1(ffi::DTYPE_I32),
        out: out.tensor3().unwrap(),
        batch_size: u32::try_from(case_count).unwrap(),
        total_tokens: u32::try_from(tokens).unwrap(),
        num_q_heads: u32::try_from(q_heads).unwrap(),
        num_k_heads: 16,
        num_v_heads: 32,
        key_dim: u32::try_from(key_dim).unwrap(),
        value_dim: 128,
        state_layout: QSCU_GDN_STATE_LAYOUT_VK,
        scale: 1.0 / (key_dim as f32).sqrt(),
        use_qk_l2norm: 1,
        disable_state_update: 0,
    };

    assert_qsfi_ok(
        unsafe { qscu_gdn_prefill(ctx.raw(), &desc) },
        "launch qwen36 GDN prefill recurrent vector",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 GDN prefill recurrent vector",
    );

    let got_out = out.to_u16_vec("download qwen36 GDN prefill recurrent output");
    assert_bf16_close_to_f32_oracle(
        "gdn prefill recurrent output",
        &got_out,
        &expected_out_bf16,
        &expected_out_f32,
        BF16_GDN_RECURRENCE_OUTPUT_ABS_TOL,
    );
    let got_state = state.to_u16_vec("download qwen36 GDN prefill final state");
    assert_bf16_close_to_f32_oracle(
        "gdn prefill recurrent final state",
        &got_state,
        &expected_state_bf16,
        &expected_state_f32,
        BF16_GDN_RECURRENCE_STATE_ABS_TOL,
    );
}

#[test]
fn qscu_gdn_decode_continuation_matches_vector() {
    let Some((_manifest, artifact)) = load_gdn_post_conv_prep_vector() else {
        return;
    };

    let seed_state_raw = artifact.tensor("recurrent_final_state_bf16").unwrap();
    let seed_state_bf16 = seed_state_raw.as_bf16_words().unwrap();
    let q1_raw = artifact.tensor("decode_step1_q_raw").unwrap();
    let k1_raw = artifact.tensor("decode_step1_k_raw").unwrap();
    let v1_raw = artifact.tensor("decode_step1_v_raw").unwrap();
    let a1_raw = artifact.tensor("decode_step1_a").unwrap();
    let b1_raw = artifact.tensor("decode_step1_b").unwrap();
    let q2_raw = artifact.tensor("decode_step2_q_raw").unwrap();
    let k2_raw = artifact.tensor("decode_step2_k_raw").unwrap();
    let v2_raw = artifact.tensor("decode_step2_v_raw").unwrap();
    let a2_raw = artifact.tensor("decode_step2_a").unwrap();
    let b2_raw = artifact.tensor("decode_step2_b").unwrap();
    let a_log_raw = artifact.tensor("A_log").unwrap();
    let dt_bias_raw = artifact.tensor("dt_bias").unwrap();
    let expected_step1_out_raw = artifact.tensor("decode_step1_output_bf16").unwrap();
    let expected_step1_out_bf16 = expected_step1_out_raw.as_bf16_words().unwrap();
    let expected_step1_out_f32 = artifact
        .tensor("decode_step1_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_step1_state_raw = artifact.tensor("decode_step1_final_state_bf16").unwrap();
    let expected_step1_state_bf16 = expected_step1_state_raw.as_bf16_words().unwrap();
    let expected_step1_state_f32 = artifact
        .tensor("decode_step1_final_state_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_step2_out_raw = artifact.tensor("decode_step2_output_bf16").unwrap();
    let expected_step2_out_bf16 = expected_step2_out_raw.as_bf16_words().unwrap();
    let expected_step2_out_f32 = artifact
        .tensor("decode_step2_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_step2_state_raw = artifact.tensor("decode_step2_final_state_bf16").unwrap();
    let expected_step2_state_bf16 = expected_step2_state_raw.as_bf16_words().unwrap();
    let expected_step2_state_f32 = artifact
        .tensor("decode_step2_final_state_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();

    let (case_count, q_heads, key_dim) = shape3(q1_raw, "decode_step1_q_raw");
    assert_eq!((q_heads, key_dim), (16, 128));
    assert_eq!(shape3(k1_raw, "decode_step1_k_raw"), (case_count, 16, 128));
    assert_eq!(shape3(v1_raw, "decode_step1_v_raw"), (case_count, 32, 128));
    assert_eq!(a1_raw.spec.shape, [case_count, 32]);
    assert_eq!(b1_raw.spec.shape, [case_count, 32]);
    assert_eq!(shape3(q2_raw, "decode_step2_q_raw"), (case_count, 16, 128));
    assert_eq!(shape3(k2_raw, "decode_step2_k_raw"), (case_count, 16, 128));
    assert_eq!(shape3(v2_raw, "decode_step2_v_raw"), (case_count, 32, 128));
    assert_eq!(a2_raw.spec.shape, [case_count, 32]);
    assert_eq!(b2_raw.spec.shape, [case_count, 32]);
    assert_eq!(a_log_raw.spec.shape, [32]);
    assert_eq!(dt_bias_raw.spec.shape, [32]);
    assert_eq!(expected_step1_out_raw.spec.shape, [case_count, 32, 128]);
    assert_eq!(expected_step2_out_raw.spec.shape, [case_count, 32, 128]);
    let (seed_cases, state_heads, state_value_dim, state_key_dim) =
        shape4(seed_state_raw, "recurrent_final_state_bf16");
    assert_eq!(seed_cases, case_count);
    assert_eq!(
        (state_heads, state_value_dim, state_key_dim),
        (32, 128, 128)
    );
    assert_eq!(
        shape4(expected_step1_state_raw, "decode_step1_final_state_bf16"),
        (case_count, 32, 128, 128)
    );
    assert_eq!(
        shape4(expected_step2_state_raw, "decode_step2_final_state_bf16"),
        (case_count, 32, 128, 128)
    );

    let state_size = state_heads * state_value_dim * state_key_dim;
    let state_pool_slots = case_count * 2;
    let mut state_pool_words = vec![0_u16; state_pool_slots * state_size];
    for case in 0..case_count {
        let src_begin = case * state_size;
        let dst_begin = (case * 2) * state_size;
        state_pool_words[dst_begin..dst_begin + state_size]
            .copy_from_slice(&seed_state_bf16[src_begin..src_begin + state_size]);
    }
    let read_slots_step1: Vec<i32> = (0..case_count)
        .map(|case| i32::try_from(case * 2).unwrap())
        .collect();
    let write_slots_step1: Vec<i32> = (0..case_count)
        .map(|case| i32::try_from(case * 2 + 1).unwrap())
        .collect();
    let read_slots_step2 = write_slots_step1.clone();
    let write_slots_step2 = read_slots_step1.clone();

    set_cuda_test_device();
    let q1 = DeviceTensor::from_bf16(q1_raw).unwrap();
    let k1 = DeviceTensor::from_bf16(k1_raw).unwrap();
    let v1 = DeviceTensor::from_bf16(v1_raw).unwrap();
    let a1 = DeviceTensor::from_bf16(a1_raw).unwrap();
    let b1 = DeviceTensor::from_bf16(b1_raw).unwrap();
    let q2 = DeviceTensor::from_bf16(q2_raw).unwrap();
    let k2 = DeviceTensor::from_bf16(k2_raw).unwrap();
    let v2 = DeviceTensor::from_bf16(v2_raw).unwrap();
    let a2 = DeviceTensor::from_bf16(a2_raw).unwrap();
    let b2 = DeviceTensor::from_bf16(b2_raw).unwrap();
    let a_log = DeviceTensor::from_f32(a_log_raw).unwrap();
    let dt_bias = DeviceTensor::from_f32(dt_bias_raw).unwrap();
    let state = DeviceTensor::<u16>::from_slice_with_dtype(
        &state_pool_words,
        vec![
            state_pool_slots,
            state_heads,
            state_value_dim,
            state_key_dim,
        ],
        Some(ffi::DTYPE_BF16),
    )
    .unwrap();
    let read1 = DeviceTensor::<i32>::from_i32_slice(&read_slots_step1, vec![case_count]).unwrap();
    let write1 = DeviceTensor::<i32>::from_i32_slice(&write_slots_step1, vec![case_count]).unwrap();
    let read2 = DeviceTensor::<i32>::from_i32_slice(&read_slots_step2, vec![case_count]).unwrap();
    let write2 = DeviceTensor::<i32>::from_i32_slice(&write_slots_step2, vec![case_count]).unwrap();
    let out1 = DeviceTensor::zeroed_bf16(expected_step1_out_raw.spec.shape.clone()).unwrap();
    let out2 = DeviceTensor::zeroed_bf16(expected_step2_out_raw.spec.shape.clone()).unwrap();
    let ctx = QsfiContext::new();

    let desc1 = QscuGdnDecodeDesc {
        q: q1.tensor3().unwrap(),
        k: k1.tensor3().unwrap(),
        v: v1.tensor3().unwrap(),
        a: a1.tensor2().unwrap(),
        b: b1.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        state: state.tensor4().unwrap(),
        state_indices: read1.tensor1().unwrap(),
        state_out_indices: write1.tensor1().unwrap(),
        out: out1.tensor3().unwrap(),
        num_tokens: u32::try_from(case_count).unwrap(),
        num_q_heads: u32::try_from(q_heads).unwrap(),
        num_k_heads: 16,
        num_v_heads: 32,
        key_dim: u32::try_from(key_dim).unwrap(),
        value_dim: 128,
        state_layout: QSCU_GDN_STATE_LAYOUT_VK,
        scale: 1.0 / (key_dim as f32).sqrt(),
        use_qk_l2norm: 1,
        disable_state_update: 0,
    };

    assert_qsfi_ok(
        unsafe { qscu_gdn_decode(ctx.raw(), &desc1) },
        "launch qwen36 GDN decode continuation step 1 vector",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 GDN decode continuation step 1 vector",
    );

    let got_step1_out = out1.to_u16_vec("download qwen36 GDN decode step 1 output");
    assert_bf16_close_to_f32_oracle(
        "gdn decode continuation step 1 output",
        &got_step1_out,
        &expected_step1_out_bf16,
        &expected_step1_out_f32,
        BF16_GDN_RECURRENCE_OUTPUT_ABS_TOL,
    );
    let state_after_step1 = state.to_u16_vec("download qwen36 GDN decode step 1 state pool");
    let got_step1_state =
        gather_compact_state_slots(&state_after_step1, &write_slots_step1, state_size);
    assert_bf16_close_to_f32_oracle(
        "gdn decode continuation step 1 final state",
        &got_step1_state,
        &expected_step1_state_bf16,
        &expected_step1_state_f32,
        BF16_GDN_RECURRENCE_STATE_ABS_TOL,
    );

    let desc2 = QscuGdnDecodeDesc {
        q: q2.tensor3().unwrap(),
        k: k2.tensor3().unwrap(),
        v: v2.tensor3().unwrap(),
        a: a2.tensor2().unwrap(),
        b: b2.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        state: state.tensor4().unwrap(),
        state_indices: read2.tensor1().unwrap(),
        state_out_indices: write2.tensor1().unwrap(),
        out: out2.tensor3().unwrap(),
        num_tokens: u32::try_from(case_count).unwrap(),
        num_q_heads: u32::try_from(q_heads).unwrap(),
        num_k_heads: 16,
        num_v_heads: 32,
        key_dim: u32::try_from(key_dim).unwrap(),
        value_dim: 128,
        state_layout: QSCU_GDN_STATE_LAYOUT_VK,
        scale: 1.0 / (key_dim as f32).sqrt(),
        use_qk_l2norm: 1,
        disable_state_update: 0,
    };

    assert_qsfi_ok(
        unsafe { qscu_gdn_decode(ctx.raw(), &desc2) },
        "launch qwen36 GDN decode continuation step 2 vector",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync qwen36 GDN decode continuation step 2 vector",
    );

    let got_step2_out = out2.to_u16_vec("download qwen36 GDN decode step 2 output");
    assert_bf16_close_to_f32_oracle(
        "gdn decode continuation step 2 output",
        &got_step2_out,
        &expected_step2_out_bf16,
        &expected_step2_out_f32,
        BF16_GDN_RECURRENCE_OUTPUT_ABS_TOL,
    );
    let state_after_step2 = state.to_u16_vec("download qwen36 GDN decode step 2 state pool");
    let got_step2_state =
        gather_compact_state_slots(&state_after_step2, &write_slots_step2, state_size);
    assert_bf16_close_to_f32_oracle(
        "gdn decode continuation step 2 final state",
        &got_step2_state,
        &expected_step2_state_bf16,
        &expected_step2_state_f32,
        BF16_GDN_RECURRENCE_STATE_ABS_TOL,
    );
}

#[test]
fn qwen36_cuda_gdn_integrated_prefill_decode_state_evolution_matches_vector() {
    let Some((_manifest, artifact)) = load_gdn_post_conv_prep_vector() else {
        return;
    };

    let mixed_qkv_raw = artifact.tensor("mixed_qkv").unwrap();
    let conv_weight_raw = artifact.tensor("conv_weight").unwrap();
    let case_offsets_raw = artifact.tensor("case_offsets").unwrap();
    let a_raw = artifact.tensor("a").unwrap();
    let b_raw = artifact.tensor("b").unwrap();
    let a_log_raw = artifact.tensor("A_log").unwrap();
    let dt_bias_raw = artifact.tensor("dt_bias").unwrap();
    let q_raw = artifact.tensor("q_raw").unwrap();
    let k_raw = artifact.tensor("k_raw").unwrap();
    let v_raw = artifact.tensor("v_raw").unwrap();
    let expected_prefill_conv_out_raw = artifact.tensor("conv_output_bf16").unwrap();
    let expected_prefill_conv_out_bf16 = expected_prefill_conv_out_raw.as_bf16_words().unwrap();
    let expected_prefill_conv_out_f32 = artifact
        .tensor("conv_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_prefill_conv_state_raw = artifact.tensor("conv_final_state").unwrap();
    let expected_prefill_conv_state = expected_prefill_conv_state_raw.as_bf16_words().unwrap();
    let expected_prefill_out_raw = artifact.tensor("recurrent_output_bf16").unwrap();
    let expected_prefill_out_bf16 = expected_prefill_out_raw.as_bf16_words().unwrap();
    let expected_prefill_out_f32 = artifact
        .tensor("recurrent_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_prefill_state_raw = artifact.tensor("recurrent_final_state_bf16").unwrap();
    let expected_prefill_state_bf16 = expected_prefill_state_raw.as_bf16_words().unwrap();
    let expected_prefill_state_f32 = artifact
        .tensor("recurrent_final_state_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();

    let decode1_mixed_raw = artifact.tensor("decode_step1_mixed_qkv").unwrap();
    let decode1_a_raw = artifact.tensor("decode_step1_a").unwrap();
    let decode1_b_raw = artifact.tensor("decode_step1_b").unwrap();
    let decode1_q_raw = artifact.tensor("decode_step1_q_raw").unwrap();
    let decode1_k_raw = artifact.tensor("decode_step1_k_raw").unwrap();
    let decode1_v_raw = artifact.tensor("decode_step1_v_raw").unwrap();
    let expected_decode1_conv_out_raw = artifact.tensor("decode_step1_conv_output_bf16").unwrap();
    let expected_decode1_conv_out_bf16 = expected_decode1_conv_out_raw.as_bf16_words().unwrap();
    let expected_decode1_conv_out_f32 = artifact
        .tensor("decode_step1_conv_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_decode1_conv_state_raw = artifact.tensor("decode_step1_conv_final_state").unwrap();
    let expected_decode1_conv_state = expected_decode1_conv_state_raw.as_bf16_words().unwrap();
    let expected_decode1_out_raw = artifact.tensor("decode_step1_output_bf16").unwrap();
    let expected_decode1_out_bf16 = expected_decode1_out_raw.as_bf16_words().unwrap();
    let expected_decode1_out_f32 = artifact
        .tensor("decode_step1_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_decode1_state_raw = artifact.tensor("decode_step1_final_state_bf16").unwrap();
    let expected_decode1_state_bf16 = expected_decode1_state_raw.as_bf16_words().unwrap();
    let expected_decode1_state_f32 = artifact
        .tensor("decode_step1_final_state_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();

    let decode2_mixed_raw = artifact.tensor("decode_step2_mixed_qkv").unwrap();
    let decode2_a_raw = artifact.tensor("decode_step2_a").unwrap();
    let decode2_b_raw = artifact.tensor("decode_step2_b").unwrap();
    let decode2_q_raw = artifact.tensor("decode_step2_q_raw").unwrap();
    let decode2_k_raw = artifact.tensor("decode_step2_k_raw").unwrap();
    let decode2_v_raw = artifact.tensor("decode_step2_v_raw").unwrap();
    let expected_decode2_conv_out_raw = artifact.tensor("decode_step2_conv_output_bf16").unwrap();
    let expected_decode2_conv_out_bf16 = expected_decode2_conv_out_raw.as_bf16_words().unwrap();
    let expected_decode2_conv_out_f32 = artifact
        .tensor("decode_step2_conv_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_decode2_conv_state_raw = artifact.tensor("decode_step2_conv_final_state").unwrap();
    let expected_decode2_conv_state = expected_decode2_conv_state_raw.as_bf16_words().unwrap();
    let expected_decode2_out_raw = artifact.tensor("decode_step2_output_bf16").unwrap();
    let expected_decode2_out_bf16 = expected_decode2_out_raw.as_bf16_words().unwrap();
    let expected_decode2_out_f32 = artifact
        .tensor("decode_step2_output_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();
    let expected_decode2_state_raw = artifact.tensor("decode_step2_final_state_bf16").unwrap();
    let expected_decode2_state_bf16 = expected_decode2_state_raw.as_bf16_words().unwrap();
    let expected_decode2_state_f32 = artifact
        .tensor("decode_step2_final_state_f32")
        .unwrap()
        .as_f32_vec()
        .unwrap();

    let (tokens, packed_dim) = shape2(mixed_qkv_raw, "mixed_qkv");
    assert_eq!(packed_dim, 8192);
    let (case_count, conv_state_packed_dim, conv_state_width) =
        shape3(expected_prefill_conv_state_raw, "conv_final_state");
    assert_eq!((conv_state_packed_dim, conv_state_width), (packed_dim, 3));
    assert_eq!(conv_weight_raw.spec.shape, [packed_dim, 4]);
    assert_eq!(case_offsets_raw.spec.shape, [case_count + 1]);
    assert_eq!(a_raw.spec.shape, [tokens, 32]);
    assert_eq!(b_raw.spec.shape, [tokens, 32]);
    assert_eq!(a_log_raw.spec.shape, [32]);
    assert_eq!(dt_bias_raw.spec.shape, [32]);
    assert_eq!(shape3(q_raw, "q_raw"), (tokens, 16, 128));
    assert_eq!(shape3(k_raw, "k_raw"), (tokens, 16, 128));
    assert_eq!(shape3(v_raw, "v_raw"), (tokens, 32, 128));
    assert_eq!(
        expected_prefill_conv_out_raw.spec.shape,
        mixed_qkv_raw.spec.shape
    );
    assert_eq!(expected_prefill_out_raw.spec.shape, [tokens, 32, 128]);

    let (state_cases, state_heads, state_value_dim, state_key_dim) =
        shape4(expected_prefill_state_raw, "recurrent_final_state_bf16");
    assert_eq!(state_cases, case_count);
    assert_eq!(
        (state_heads, state_value_dim, state_key_dim),
        (32, 128, 128)
    );
    assert_eq!(
        shape4(expected_decode1_state_raw, "decode_step1_final_state_bf16"),
        (case_count, 32, 128, 128)
    );
    assert_eq!(
        shape4(expected_decode2_state_raw, "decode_step2_final_state_bf16"),
        (case_count, 32, 128, 128)
    );

    assert_eq!(decode1_mixed_raw.spec.shape, [case_count, packed_dim]);
    assert_eq!(decode2_mixed_raw.spec.shape, [case_count, packed_dim]);
    assert_eq!(decode1_a_raw.spec.shape, [case_count, 32]);
    assert_eq!(decode1_b_raw.spec.shape, [case_count, 32]);
    assert_eq!(decode2_a_raw.spec.shape, [case_count, 32]);
    assert_eq!(decode2_b_raw.spec.shape, [case_count, 32]);
    assert_eq!(
        shape3(decode1_q_raw, "decode_step1_q_raw"),
        (case_count, 16, 128)
    );
    assert_eq!(
        shape3(decode1_k_raw, "decode_step1_k_raw"),
        (case_count, 16, 128)
    );
    assert_eq!(
        shape3(decode1_v_raw, "decode_step1_v_raw"),
        (case_count, 32, 128)
    );
    assert_eq!(
        shape3(decode2_q_raw, "decode_step2_q_raw"),
        (case_count, 16, 128)
    );
    assert_eq!(
        shape3(decode2_k_raw, "decode_step2_k_raw"),
        (case_count, 16, 128)
    );
    assert_eq!(
        shape3(decode2_v_raw, "decode_step2_v_raw"),
        (case_count, 32, 128)
    );
    assert_eq!(
        expected_decode1_conv_out_raw.spec.shape,
        [case_count, packed_dim]
    );
    assert_eq!(
        expected_decode2_conv_out_raw.spec.shape,
        [case_count, packed_dim]
    );
    assert_eq!(
        shape3(
            expected_decode1_conv_state_raw,
            "decode_step1_conv_final_state"
        ),
        (case_count, packed_dim, conv_state_width)
    );
    assert_eq!(
        shape3(
            expected_decode2_conv_state_raw,
            "decode_step2_conv_final_state"
        ),
        (case_count, packed_dim, conv_state_width)
    );
    assert_eq!(expected_decode1_out_raw.spec.shape, [case_count, 32, 128]);
    assert_eq!(expected_decode2_out_raw.spec.shape, [case_count, 32, 128]);

    let state_pool_slots = case_count * 2;
    let slot0 = case_ping_pong_slots(case_count, 0);
    let slot1 = case_ping_pong_slots(case_count, 1);
    let conv_state_size = packed_dim * conv_state_width;
    let recurrent_state_size = state_heads * state_value_dim * state_key_dim;

    set_cuda_test_device();
    let mixed_qkv = DeviceTensor::from_bf16(mixed_qkv_raw).unwrap();
    let conv_weight = DeviceTensor::from_bf16(conv_weight_raw).unwrap();
    let a = DeviceTensor::from_bf16(a_raw).unwrap();
    let b = DeviceTensor::from_bf16(b_raw).unwrap();
    let a_log = DeviceTensor::from_f32(a_log_raw).unwrap();
    let dt_bias = DeviceTensor::from_f32(dt_bias_raw).unwrap();
    let seq_indptr = DeviceTensor::from_i32(case_offsets_raw).unwrap();
    let slot0_indices = DeviceTensor::<i32>::from_i32_slice(&slot0, vec![case_count]).unwrap();
    let slot1_indices = DeviceTensor::<i32>::from_i32_slice(&slot1, vec![case_count]).unwrap();
    let no_state_read =
        DeviceTensor::<i32>::from_i32_slice(&vec![-1_i32; case_count], vec![case_count]).unwrap();
    let conv_state =
        DeviceTensor::zeroed_bf16(vec![state_pool_slots, packed_dim, conv_state_width]).unwrap();
    let prefill_conv_out =
        DeviceTensor::zeroed_bf16(expected_prefill_conv_out_raw.spec.shape.clone()).unwrap();

    let prefill_conv_desc = QscuQwen36GdnCausalConv1dDesc {
        x: mixed_qkv.tensor2().unwrap(),
        weight: conv_weight.tensor2().unwrap(),
        bias: absent_tensor1(ffi::DTYPE_BF16),
        state: conv_state.tensor3().unwrap(),
        state_read_indices: no_state_read.tensor1().unwrap(),
        state_write_indices: slot0_indices.tensor1().unwrap(),
        seq_indptr: seq_indptr.as_device_ptr(),
        out: prefill_conv_out.tensor2().unwrap(),
        num_tokens: u32::try_from(tokens).unwrap(),
        batch_size: u32::try_from(case_count).unwrap(),
        activation: QSCU_ACTIVATION_SILU,
        update_state: 1,
    };
    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_causal_conv1d_bf16(&prefill_conv_desc, ptr::null_mut()) },
        "launch integrated qwen36 GDN prefill conv",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync integrated prefill conv",
    );

    let got_prefill_conv = prefill_conv_out.to_u16_vec("download integrated prefill conv output");
    assert_bf16_close_to_f32_oracle(
        "integrated gdn prefill conv output",
        &got_prefill_conv,
        &expected_prefill_conv_out_bf16,
        &expected_prefill_conv_out_f32,
        BF16_GDN_CONV_ABS_TOL,
    );
    let conv_state_after_prefill =
        conv_state.to_u16_vec("download integrated prefill conv state pool");
    let got_prefill_conv_state =
        gather_compact_state_slots(&conv_state_after_prefill, &slot0, conv_state_size);
    assert_u16_words_eq(
        "integrated gdn prefill conv final state",
        &got_prefill_conv_state,
        &expected_prefill_conv_state,
    );

    let q_prefill = DeviceTensor::zeroed_bf16(q_raw.spec.shape.clone()).unwrap();
    let k_prefill = DeviceTensor::zeroed_bf16(k_raw.spec.shape.clone()).unwrap();
    let v_prefill = DeviceTensor::zeroed_bf16(v_raw.spec.shape.clone()).unwrap();
    let prefill_post_desc = QscuQwen36GdnPostConvPrepareDesc {
        conv_out: prefill_conv_out.tensor2().unwrap(),
        a: a.tensor2().unwrap(),
        b: b.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        q: q_prefill.tensor3().unwrap(),
        k: k_prefill.tensor3().unwrap(),
        v: v_prefill.tensor3().unwrap(),
        g_out: absent_tensor2(ffi::DTYPE_F32),
        beta_out: absent_tensor2(ffi::DTYPE_F32),
        num_tokens: u32::try_from(tokens).unwrap(),
        apply_qk_l2norm: 0,
        l2norm_eps: 1.0e-6,
        forget_gate_output: QSCU_GDN_FORGET_LOG_DECAY,
    };
    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_post_conv_prepare_bf16(&prefill_post_desc, ptr::null_mut()) },
        "launch integrated qwen36 GDN prefill post-conv prep",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync integrated prefill post-conv prep",
    );

    let recurrent_state = DeviceTensor::zeroed_bf16(vec![
        state_pool_slots,
        state_heads,
        state_value_dim,
        state_key_dim,
    ])
    .unwrap();
    let prefill_out =
        DeviceTensor::zeroed_bf16(expected_prefill_out_raw.spec.shape.clone()).unwrap();
    let ctx = QsfiContext::new();
    let prefill_desc = QscuGdnPrefillDesc {
        q: q_prefill.tensor3().unwrap(),
        k: k_prefill.tensor3().unwrap(),
        v: v_prefill.tensor3().unwrap(),
        a: a.tensor2().unwrap(),
        b: b.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        state: recurrent_state.tensor4().unwrap(),
        seq_indptr: seq_indptr.as_device_ptr(),
        state_indices: slot0_indices.tensor1().unwrap(),
        state_out_indices: slot0_indices.tensor1().unwrap(),
        out: prefill_out.tensor3().unwrap(),
        batch_size: u32::try_from(case_count).unwrap(),
        total_tokens: u32::try_from(tokens).unwrap(),
        num_q_heads: 16,
        num_k_heads: 16,
        num_v_heads: 32,
        key_dim: 128,
        value_dim: 128,
        state_layout: QSCU_GDN_STATE_LAYOUT_VK,
        scale: 1.0 / 128.0_f32.sqrt(),
        use_qk_l2norm: 1,
        disable_state_update: 0,
    };
    assert_qsfi_ok(
        unsafe { qscu_gdn_prefill(ctx.raw(), &prefill_desc) },
        "launch integrated qwen36 GDN prefill recurrence",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync integrated prefill recurrence",
    );

    let got_prefill_out = prefill_out.to_u16_vec("download integrated prefill recurrent output");
    assert_bf16_close_to_f32_oracle(
        "integrated gdn prefill recurrent output",
        &got_prefill_out,
        &expected_prefill_out_bf16,
        &expected_prefill_out_f32,
        BF16_GDN_RECURRENCE_OUTPUT_ABS_TOL,
    );
    let recurrent_state_after_prefill =
        recurrent_state.to_u16_vec("download integrated prefill recurrent state pool");
    let got_prefill_state =
        gather_compact_state_slots(&recurrent_state_after_prefill, &slot0, recurrent_state_size);
    assert_bf16_close_to_f32_oracle(
        "integrated gdn prefill recurrent final state",
        &got_prefill_state,
        &expected_prefill_state_bf16,
        &expected_prefill_state_f32,
        BF16_GDN_RECURRENCE_STATE_ABS_TOL,
    );

    let decode1_mixed = DeviceTensor::from_bf16(decode1_mixed_raw).unwrap();
    let decode1_conv_out =
        DeviceTensor::zeroed_bf16(expected_decode1_conv_out_raw.spec.shape.clone()).unwrap();
    let decode1_conv_desc = QscuQwen36GdnCausalConv1dDesc {
        x: decode1_mixed.tensor2().unwrap(),
        weight: conv_weight.tensor2().unwrap(),
        bias: absent_tensor1(ffi::DTYPE_BF16),
        state: conv_state.tensor3().unwrap(),
        state_read_indices: slot0_indices.tensor1().unwrap(),
        state_write_indices: slot1_indices.tensor1().unwrap(),
        seq_indptr: ptr::null_mut(),
        out: decode1_conv_out.tensor2().unwrap(),
        num_tokens: u32::try_from(case_count).unwrap(),
        batch_size: u32::try_from(case_count).unwrap(),
        activation: QSCU_ACTIVATION_SILU,
        update_state: 1,
    };
    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_causal_conv1d_bf16(&decode1_conv_desc, ptr::null_mut()) },
        "launch integrated qwen36 GDN decode step 1 conv",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync integrated decode step 1 conv",
    );
    let got_decode1_conv = decode1_conv_out.to_u16_vec("download integrated decode step 1 conv");
    assert_bf16_close_to_f32_oracle(
        "integrated gdn decode step 1 conv output",
        &got_decode1_conv,
        &expected_decode1_conv_out_bf16,
        &expected_decode1_conv_out_f32,
        BF16_GDN_CONV_ABS_TOL,
    );
    let conv_state_after_decode1 =
        conv_state.to_u16_vec("download integrated decode step 1 conv state pool");
    let got_decode1_conv_state =
        gather_compact_state_slots(&conv_state_after_decode1, &slot1, conv_state_size);
    assert_u16_words_eq(
        "integrated gdn decode step 1 conv final state",
        &got_decode1_conv_state,
        &expected_decode1_conv_state,
    );

    let decode1_a = DeviceTensor::from_bf16(decode1_a_raw).unwrap();
    let decode1_b = DeviceTensor::from_bf16(decode1_b_raw).unwrap();
    let q1 = DeviceTensor::zeroed_bf16(decode1_q_raw.spec.shape.clone()).unwrap();
    let k1 = DeviceTensor::zeroed_bf16(decode1_k_raw.spec.shape.clone()).unwrap();
    let v1 = DeviceTensor::zeroed_bf16(decode1_v_raw.spec.shape.clone()).unwrap();
    let decode1_post_desc = QscuQwen36GdnPostConvPrepareDesc {
        conv_out: decode1_conv_out.tensor2().unwrap(),
        a: decode1_a.tensor2().unwrap(),
        b: decode1_b.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        q: q1.tensor3().unwrap(),
        k: k1.tensor3().unwrap(),
        v: v1.tensor3().unwrap(),
        g_out: absent_tensor2(ffi::DTYPE_F32),
        beta_out: absent_tensor2(ffi::DTYPE_F32),
        num_tokens: u32::try_from(case_count).unwrap(),
        apply_qk_l2norm: 0,
        l2norm_eps: 1.0e-6,
        forget_gate_output: QSCU_GDN_FORGET_LOG_DECAY,
    };
    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_post_conv_prepare_bf16(&decode1_post_desc, ptr::null_mut()) },
        "launch integrated qwen36 GDN decode step 1 post-conv prep",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync integrated decode step 1 post-conv prep",
    );

    let decode1_out =
        DeviceTensor::zeroed_bf16(expected_decode1_out_raw.spec.shape.clone()).unwrap();
    let decode1_desc = QscuGdnDecodeDesc {
        q: q1.tensor3().unwrap(),
        k: k1.tensor3().unwrap(),
        v: v1.tensor3().unwrap(),
        a: decode1_a.tensor2().unwrap(),
        b: decode1_b.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        state: recurrent_state.tensor4().unwrap(),
        state_indices: slot0_indices.tensor1().unwrap(),
        state_out_indices: slot1_indices.tensor1().unwrap(),
        out: decode1_out.tensor3().unwrap(),
        num_tokens: u32::try_from(case_count).unwrap(),
        num_q_heads: 16,
        num_k_heads: 16,
        num_v_heads: 32,
        key_dim: 128,
        value_dim: 128,
        state_layout: QSCU_GDN_STATE_LAYOUT_VK,
        scale: 1.0 / 128.0_f32.sqrt(),
        use_qk_l2norm: 1,
        disable_state_update: 0,
    };
    assert_qsfi_ok(
        unsafe { qscu_gdn_decode(ctx.raw(), &decode1_desc) },
        "launch integrated qwen36 GDN decode step 1 recurrence",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync integrated decode step 1 recurrence",
    );
    let got_decode1_out = decode1_out.to_u16_vec("download integrated decode step 1 output");
    assert_bf16_close_to_f32_oracle(
        "integrated gdn decode step 1 recurrent output",
        &got_decode1_out,
        &expected_decode1_out_bf16,
        &expected_decode1_out_f32,
        BF16_GDN_RECURRENCE_OUTPUT_ABS_TOL,
    );
    let recurrent_state_after_decode1 =
        recurrent_state.to_u16_vec("download integrated decode step 1 recurrent state pool");
    let got_decode1_state =
        gather_compact_state_slots(&recurrent_state_after_decode1, &slot1, recurrent_state_size);
    assert_bf16_close_to_f32_oracle(
        "integrated gdn decode step 1 recurrent final state",
        &got_decode1_state,
        &expected_decode1_state_bf16,
        &expected_decode1_state_f32,
        BF16_GDN_RECURRENCE_STATE_ABS_TOL,
    );

    let decode2_mixed = DeviceTensor::from_bf16(decode2_mixed_raw).unwrap();
    let decode2_conv_out =
        DeviceTensor::zeroed_bf16(expected_decode2_conv_out_raw.spec.shape.clone()).unwrap();
    let decode2_conv_desc = QscuQwen36GdnCausalConv1dDesc {
        x: decode2_mixed.tensor2().unwrap(),
        weight: conv_weight.tensor2().unwrap(),
        bias: absent_tensor1(ffi::DTYPE_BF16),
        state: conv_state.tensor3().unwrap(),
        state_read_indices: slot1_indices.tensor1().unwrap(),
        state_write_indices: slot0_indices.tensor1().unwrap(),
        seq_indptr: ptr::null_mut(),
        out: decode2_conv_out.tensor2().unwrap(),
        num_tokens: u32::try_from(case_count).unwrap(),
        batch_size: u32::try_from(case_count).unwrap(),
        activation: QSCU_ACTIVATION_SILU,
        update_state: 1,
    };
    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_causal_conv1d_bf16(&decode2_conv_desc, ptr::null_mut()) },
        "launch integrated qwen36 GDN decode step 2 conv",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync integrated decode step 2 conv",
    );
    let got_decode2_conv = decode2_conv_out.to_u16_vec("download integrated decode step 2 conv");
    assert_bf16_close_to_f32_oracle(
        "integrated gdn decode step 2 conv output",
        &got_decode2_conv,
        &expected_decode2_conv_out_bf16,
        &expected_decode2_conv_out_f32,
        BF16_GDN_CONV_ABS_TOL,
    );
    let conv_state_after_decode2 =
        conv_state.to_u16_vec("download integrated decode step 2 conv state pool");
    let got_decode2_conv_state =
        gather_compact_state_slots(&conv_state_after_decode2, &slot0, conv_state_size);
    assert_u16_words_eq(
        "integrated gdn decode step 2 conv final state",
        &got_decode2_conv_state,
        &expected_decode2_conv_state,
    );

    let decode2_a = DeviceTensor::from_bf16(decode2_a_raw).unwrap();
    let decode2_b = DeviceTensor::from_bf16(decode2_b_raw).unwrap();
    let q2 = DeviceTensor::zeroed_bf16(decode2_q_raw.spec.shape.clone()).unwrap();
    let k2 = DeviceTensor::zeroed_bf16(decode2_k_raw.spec.shape.clone()).unwrap();
    let v2 = DeviceTensor::zeroed_bf16(decode2_v_raw.spec.shape.clone()).unwrap();
    let decode2_post_desc = QscuQwen36GdnPostConvPrepareDesc {
        conv_out: decode2_conv_out.tensor2().unwrap(),
        a: decode2_a.tensor2().unwrap(),
        b: decode2_b.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        q: q2.tensor3().unwrap(),
        k: k2.tensor3().unwrap(),
        v: v2.tensor3().unwrap(),
        g_out: absent_tensor2(ffi::DTYPE_F32),
        beta_out: absent_tensor2(ffi::DTYPE_F32),
        num_tokens: u32::try_from(case_count).unwrap(),
        apply_qk_l2norm: 0,
        l2norm_eps: 1.0e-6,
        forget_gate_output: QSCU_GDN_FORGET_LOG_DECAY,
    };
    assert_qsfi_ok(
        unsafe { qscu_qwen36_gdn_post_conv_prepare_bf16(&decode2_post_desc, ptr::null_mut()) },
        "launch integrated qwen36 GDN decode step 2 post-conv prep",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync integrated decode step 2 post-conv prep",
    );

    let decode2_out =
        DeviceTensor::zeroed_bf16(expected_decode2_out_raw.spec.shape.clone()).unwrap();
    let decode2_desc = QscuGdnDecodeDesc {
        q: q2.tensor3().unwrap(),
        k: k2.tensor3().unwrap(),
        v: v2.tensor3().unwrap(),
        a: decode2_a.tensor2().unwrap(),
        b: decode2_b.tensor2().unwrap(),
        a_log: a_log.tensor1().unwrap(),
        dt_bias: dt_bias.tensor1().unwrap(),
        state: recurrent_state.tensor4().unwrap(),
        state_indices: slot1_indices.tensor1().unwrap(),
        state_out_indices: slot0_indices.tensor1().unwrap(),
        out: decode2_out.tensor3().unwrap(),
        num_tokens: u32::try_from(case_count).unwrap(),
        num_q_heads: 16,
        num_k_heads: 16,
        num_v_heads: 32,
        key_dim: 128,
        value_dim: 128,
        state_layout: QSCU_GDN_STATE_LAYOUT_VK,
        scale: 1.0 / 128.0_f32.sqrt(),
        use_qk_l2norm: 1,
        disable_state_update: 0,
    };
    assert_qsfi_ok(
        unsafe { qscu_gdn_decode(ctx.raw(), &decode2_desc) },
        "launch integrated qwen36 GDN decode step 2 recurrence",
    );
    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync integrated decode step 2 recurrence",
    );
    let got_decode2_out = decode2_out.to_u16_vec("download integrated decode step 2 output");
    assert_bf16_close_to_f32_oracle(
        "integrated gdn decode step 2 recurrent output",
        &got_decode2_out,
        &expected_decode2_out_bf16,
        &expected_decode2_out_f32,
        BF16_GDN_RECURRENCE_OUTPUT_ABS_TOL,
    );
    let recurrent_state_after_decode2 =
        recurrent_state.to_u16_vec("download integrated decode step 2 recurrent state pool");
    let got_decode2_state =
        gather_compact_state_slots(&recurrent_state_after_decode2, &slot0, recurrent_state_size);
    assert_bf16_close_to_f32_oracle(
        "integrated gdn decode step 2 recurrent final state",
        &got_decode2_state,
        &expected_decode2_state_bf16,
        &expected_decode2_state_f32,
        BF16_GDN_RECURRENCE_STATE_ABS_TOL,
    );
}

#[test]
fn moe_vector_exposes_router_and_topk_views_when_present() {
    let root = required_vector_root();
    let manifest = root.join("moe_shared_expert/manifest.json");
    require_manifest(&manifest);

    let artifact = VectorArtifact::load(manifest).unwrap();
    let router_logits = artifact.tensor("router_logits").unwrap();
    let topk_ids = artifact.tensor("topk_ids").unwrap();
    let topk_weights = artifact.tensor("topk_weights").unwrap();

    assert_eq!(router_logits.spec.shape, [5, 256]);
    assert_eq!(router_logits.as_f32_vec().unwrap().len(), 5 * 256);
    assert_eq!(topk_ids.spec.shape, [5, 8]);
    assert_eq!(topk_ids.as_i32_vec().unwrap().len(), 5 * 8);
    assert_eq!(topk_weights.spec.shape, [5, 8]);
    assert_eq!(topk_weights.as_f32_vec().unwrap().len(), 5 * 8);
}
