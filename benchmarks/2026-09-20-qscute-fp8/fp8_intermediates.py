"""Record real QKV disagreements while feeding cuBLASLt outputs onward."""
from pathlib import Path
import fp8_model_parity as parity

original_copytree = parity.shutil.copytree
PATCH = r'''
                    let mut candidate = crate::memory::HostBuffer::<BF16>::new(PACKED_QKV_CHANNELS as usize)?;
                    scratch.packed.download_range(0, &mut candidate)?;
                    ctx.synchronize()?;
                    ops.qscb().linear_fp8(x, weight, packed, [input_scale, weight_scale], self.linear_workspace)?;
                    let mut reference = crate::memory::HostBuffer::<BF16>::new(PACKED_QKV_CHANNELS as usize)?;
                    scratch.packed.download_range(0, &mut reference)?;
                    ctx.synchronize()?;
                    let decode = |bytes: &[u8]| -> Vec<f32> { bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16)).collect() };
                    let a = decode(candidate.as_ref()); let b = decode(reference.as_ref());
                    let changed = a.iter().zip(&b).filter(|(a,b)| a != b).count();
                    let error:f64 = a.iter().zip(&b).map(|(a,b)| f64::from(a-b).powi(2)).sum();
                    let norm:f64 = b.iter().map(|v| f64::from(*v).powi(2)).sum();
                    let max = a.iter().zip(&b).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
                    eprintln!("QKV {{\"layer\":{gdn_layer_idx},\"changed\":{changed},\"max_abs\":{max},\"relative_l2\":{}}}", (error/norm).sqrt());
                    let root=std::path::PathBuf::from(std::env::var_os("KR03_OUTPUT").unwrap());
                    std::fs::write(root.join(format!("qkv-{gdn_layer_idx}.cute.bf16")),candidate.as_ref()).unwrap();
                    std::fs::write(root.join(format!("qkv-{gdn_layer_idx}.reference.bf16")),reference.as_ref()).unwrap();
                    if gdn_layer_idx < 3 {
                        let mut h = crate::memory::HostBuffer::<crate::dtype::Fp8E4M3>::new(hidden as usize)?;
                        quantized.fp8.download_range(0,&mut h)?;
                        let mut w = crate::memory::HostBuffer::<crate::dtype::Fp8E4M3>::new((hidden*PACKED_QKV_CHANNELS) as usize)?;
                        ctx.download(weight.data.as_raw(),w.as_mut())?;
                        let mut xs = crate::memory::HostBuffer::<crate::dtype::F32>::new(1)?;
                        let mut ws = crate::memory::HostBuffer::<crate::dtype::F32>::new(1)?;
                        ctx.download(input_scale.data.as_raw(),xs.as_mut())?;
                        ctx.download(weight_scale.data.as_raw(),ws.as_mut())?;
                        ctx.synchronize()?;
                        std::fs::write(root.join(format!("qkv-{gdn_layer_idx}.x.fp8")),h.as_ref()).unwrap();
                        std::fs::write(root.join(format!("qkv-{gdn_layer_idx}.w.fp8")),w.as_ref()).unwrap();
                        std::fs::write(root.join(format!("qkv-{gdn_layer_idx}.scales.f32")),[xs.as_ref(),ws.as_ref()].concat()).unwrap();
                    }
'''
def copytree(source, dest, *args, **kw):
    result=original_copytree(source, dest, *args, **kw)
    if Path(source)==parity.ROOT/'src':
        path = Path(dest) / 'model/runner/gdn.rs'
        source_text = path.read_text()
        needle = '                    reduce.launch(ctx.stream, partials, packed)?;'
        assert source_text.count(needle) == 1, 'expected the runner-owned FP8 reduction'
        path.write_text(source_text.replace(needle, needle + PATCH))
        path=Path(dest)/'backend/tensor.rs';path.write_text(path.read_text().replace('pub(super) data:', 'pub(crate) data:'))
    return result

parity.shutil.copytree=copytree
parity.TEST=parity.TEST.replace('[4usize, 102, 1024]','[4usize]').replace('0..=32','0..=1').replace('step < 32','step < 1')
parity.main(expected_rows=2, protocol='One runner, four-token prefix and one forced decode; compare identical-input QKV at all 48 GDN layers and feed cuBLASLt outputs onward. Diagnostic downloads invalidate timing.')
