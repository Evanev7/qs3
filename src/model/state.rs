use super::{QwenConfig, checked_usize_product};
use crate::{
    QWEN36_GDN_STATE_SLOTS_PER_LAYER,
    backend::{DMat, FloatStorage, GdnConvState, GdnRecurrentState},
    constants::{
        GdnRecurrentDType,
        gdn::{
            CONV_HISTORY_LEN, KEY_HEAD_DIM, NUM_VALUE_HEADS, PACKED_QKV_CHANNELS, VALUE_HEAD_DIM,
        },
        precision::GDN_RECURRENT_STATE,
    },
    dtype::{BF16, DType},
    engine::Status,
    ext::SafeVec,
    ffi::DevicePtr,
    memory::{CudaCtx, DeviceBuffer},
};
use std::{mem, rc::Rc};

const GDN_RECURRENT_STORAGE: FloatStorage = match GDN_RECURRENT_STATE.as_bytes() {
    b"bf16" => FloatStorage::Bf16,
    b"f32" => FloatStorage::F32,
    _ => panic!("unsupported compiled GDN recurrent storage"),
};

pub(super) struct GdnLayerSlots {
    pub(super) live_slot: u32,
    pub(super) staged_slot: u32,
}

pub(super) struct GdnSlotMap {
    pub(super) live_slots: Vec<u32>,
    pub(super) staged_slots: Vec<u32>,
    pub(super) state_pool: u32,
}

impl GdnSlotMap {
    pub(super) fn new(gdn_layer_count: u32) -> Result<Self, Status> {
        let state_pool = gdn_layer_count
            .checked_mul(QWEN36_GDN_STATE_SLOTS_PER_LAYER)
            .ok_or(Status::InvalidArgument)?;
        let mut live_slots = Vec::safe_new(gdn_layer_count as usize)?;
        let mut staged_slots = Vec::safe_new(gdn_layer_count as usize)?;
        Self::reset_slots(gdn_layer_count, &mut live_slots, &mut staged_slots)?;
        Ok(Self {
            live_slots,
            staged_slots,
            state_pool,
        })
    }

    pub(super) fn reset(&mut self, gdn_layer_count: u32) -> Result<(), Status> {
        if gdn_layer_count as usize != self.live_slots.len() {
            return Err(Status::InternalError);
        }
        Self::reset_slots(
            gdn_layer_count,
            &mut self.live_slots,
            &mut self.staged_slots,
        )
    }

    pub(super) fn reset_slots(
        gdn_layer_count: u32,
        live_slots: &mut Vec<u32>,
        staged_slots: &mut Vec<u32>,
    ) -> Result<(), Status> {
        live_slots.clear();
        staged_slots.clear();
        for gdn_layer_idx in 0..gdn_layer_count {
            let base = gdn_layer_idx
                .checked_mul(QWEN36_GDN_STATE_SLOTS_PER_LAYER)
                .ok_or(Status::InvalidArgument)?;
            live_slots.push(base);
            staged_slots.push(base.checked_add(1).ok_or(Status::InvalidArgument)?);
        }
        Ok(())
    }

    pub(super) fn layer_slots(&self, gdn_layer_idx: u32) -> Result<GdnLayerSlots, Status> {
        let idx = gdn_layer_idx as usize;
        let live_slot = *self.live_slots.get(idx).ok_or(Status::InvalidArgument)?;
        let staged_slot = *self.staged_slots.get(idx).ok_or(Status::InvalidArgument)?;
        if live_slot >= self.state_pool || staged_slot >= self.state_pool {
            return Err(Status::InternalError);
        }
        Ok(GdnLayerSlots {
            live_slot,
            staged_slot,
        })
    }

    pub(super) fn commit(&mut self) {
        for idx in 0..self.live_slots.len() {
            mem::swap(&mut self.live_slots[idx], &mut self.staged_slots[idx]);
        }
    }
}

pub(super) struct GdnState {
    pub(super) conv: DeviceBuffer<BF16>,
    recurrent: DeviceBuffer<GdnRecurrentDType>,
    pub(super) slots: GdnSlotMap,
}

impl GdnState {
    pub(super) fn new(ctx: Rc<CudaCtx>, config: &QwenConfig) -> Result<Self, Status> {
        let slots = GdnSlotMap::new(config.gdn_layer_count())?;
        let state_pool = slots.state_pool;
        let conv_len = checked_usize_product(&[state_pool, PACKED_QKV_CHANNELS, CONV_HISTORY_LEN])?;
        let recurrent_len =
            checked_usize_product(&[state_pool, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM])?;
        let mut state = Self {
            conv: DeviceBuffer::with_capacity(ctx.clone(), conv_len)?,
            recurrent: DeviceBuffer::with_capacity(ctx.clone(), recurrent_len)?,
            slots,
        };
        state.zero()?;
        Ok(state)
    }

    pub(super) fn reset(&mut self, config: &QwenConfig) -> Result<(), Status> {
        self.slots.reset(config.gdn_layer_count())?;
        self.zero()
    }

    pub(super) fn zero(&mut self) -> Result<(), Status> {
        self.conv.zero()?;
        self.recurrent.zero()
    }

    pub(super) fn conv_view(&self) -> Result<GdnConvState, Status> {
        self.conv
            .tensor3(self.slots.state_pool, PACKED_QKV_CHANNELS, CONV_HISTORY_LEN)?;
        GdnConvState::contiguous(self.conv.erase(), FloatStorage::Bf16, self.slots.state_pool)
    }

    pub(super) fn recurrent_slot(&self, slot: u32) -> Result<DMat<GdnRecurrentDType>, Status> {
        if slot >= self.slots.state_pool {
            return Err(Status::InvalidArgument);
        }
        let offset = checked_usize_product(&[slot, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM])?;
        let bytes = GdnRecurrentDType::size_of(offset)?;
        // The slot is within the owned, contiguous state pool; no device metadata is read.
        let ptr = unsafe { self.recurrent.as_raw().add(bytes) };
        DMat::contiguous(
            DevicePtr::new(ptr).ok_or(Status::InvalidArgument)?,
            NUM_VALUE_HEADS * VALUE_HEAD_DIM,
            KEY_HEAD_DIM,
        )
    }

    pub(super) fn recurrent_view(&self) -> Result<GdnRecurrentState, Status> {
        GdnRecurrentState::contiguous(
            self.recurrent.erase(),
            GDN_RECURRENT_STORAGE,
            self.slots.state_pool,
        )
    }

    pub(super) fn layer_slots(&self, gdn_layer_idx: u32) -> Result<GdnLayerSlots, Status> {
        self.slots.layer_slots(gdn_layer_idx)
    }

    pub(super) fn commit(&mut self) {
        self.slots.commit();
    }
}
