use super::{GdnRecurrentPrecision, QwenConfig, checked_usize_product, scratch::DeviceBuffer};
use crate::{
    QWEN36_GDN_CONV_STATE, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_PACKED_DIM,
    QWEN36_GDN_STATE_SLOTS_PER_LAYER, QWEN36_GDN_VALUE_DIM, engine::Status, ext::SafeVec,
};

use crate::backend::{FloatStorage, GdnRecurrentState};
use std::mem;

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

enum RecurrentBuffer {
    Bf16(DeviceBuffer<u16>),
    F32(DeviceBuffer<f32>),
}

impl RecurrentBuffer {
    fn new(config: &QwenConfig, elements: usize) -> Result<Self, Status> {
        match config.gdn_recurrent_precision {
            GdnRecurrentPrecision::Bf16 => {
                let mut buffer = DeviceBuffer::empty(config.device_ordinal);
                buffer.ensure(elements)?;
                Ok(Self::Bf16(buffer))
            }
            GdnRecurrentPrecision::F32 => {
                let mut buffer = DeviceBuffer::empty(config.device_ordinal);
                buffer.ensure(elements)?;
                Ok(Self::F32(buffer))
            }
        }
    }

    fn zero(&mut self, stream: crate::ffi::CudaStream) -> Result<(), Status> {
        match self {
            Self::Bf16(buffer) => buffer.zero(buffer.cap, stream),
            Self::F32(buffer) => buffer.zero(buffer.cap, stream),
        }
    }

    fn view(&self, state_pool: u32) -> Result<GdnRecurrentState, Status> {
        let (data, dtype) = match self {
            Self::Bf16(buffer) => (buffer.as_device_ptr(), FloatStorage::Bf16),
            Self::F32(buffer) => (buffer.as_device_ptr(), FloatStorage::F32),
        };
        GdnRecurrentState::contiguous(data, dtype, state_pool)
    }
}

pub(super) struct GdnState {
    pub(super) conv: DeviceBuffer<u16>,
    recurrent: RecurrentBuffer,
    pub(super) slots: GdnSlotMap,
}

impl GdnState {
    pub(super) fn new(config: &QwenConfig) -> Result<Self, Status> {
        let slots = GdnSlotMap::new(config.gdn_layer_count())?;
        let state_pool = slots.state_pool;
        let conv_len =
            checked_usize_product(&[state_pool, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_CONV_STATE])?;
        let recurrent_len = checked_usize_product(&[
            state_pool,
            QWEN36_GDN_NUM_V_HEADS,
            QWEN36_GDN_VALUE_DIM,
            QWEN36_GDN_KEY_DIM,
        ])?;
        let mut state = Self {
            conv: DeviceBuffer::empty(config.device_ordinal),
            recurrent: RecurrentBuffer::new(config, recurrent_len)?,
            slots,
        };
        state.conv.ensure(conv_len)?;
        state.zero(config)?;
        Ok(state)
    }

    pub(super) fn reset(&mut self, config: &QwenConfig) -> Result<(), Status> {
        self.slots.reset(config.gdn_layer_count())?;
        self.zero(config)
    }

    pub(super) fn zero(&mut self, config: &QwenConfig) -> Result<(), Status> {
        self.conv.zero(self.conv.cap, config.stream)?;
        self.recurrent.zero(config.stream)
    }

    pub(super) fn recurrent_view(&self) -> Result<GdnRecurrentState, Status> {
        self.recurrent.view(self.slots.state_pool)
    }

    pub(super) fn layer_slots(&self, gdn_layer_idx: u32) -> Result<GdnLayerSlots, Status> {
        self.slots.layer_slots(gdn_layer_idx)
    }

    pub(super) fn commit(&mut self) {
        self.slots.commit();
    }
}
