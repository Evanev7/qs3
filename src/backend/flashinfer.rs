use crate::{Status, ffi::qsfi};

pub(crate) use super::{
    FusedAddRmsNormBf16, GdnDecodeBf16, GdnPrefillBf16, MoeBf16Execute, MoeBf16ExecuteArgs,
    MoeBf16PlanConfig, MoePlan, RmsNormBf16, RopeApplyBf16,
};

pub(crate) use super::{GdnDecodeBf16Args, GdnPrefillBf16Args, Workspace};

/// Thin typed access to qsfi's FlashInfer-owned operations.
pub(crate) struct FlashInfer<'a> {
    context: &'a mut qsfi::Context,
}

impl<'a> FlashInfer<'a> {
    pub(super) fn new(context: &'a mut qsfi::Context) -> Self {
        Self { context }
    }

    pub(crate) unsafe fn rmsnorm_bf16(&mut self, desc: &RmsNormBf16) -> Result<(), Status> {
        unsafe { self.context.rmsnorm(&desc.raw) }
    }

    pub(crate) unsafe fn fused_add_rmsnorm_bf16(
        &mut self,
        desc: &FusedAddRmsNormBf16,
    ) -> Result<(), Status> {
        unsafe { self.context.fused_add_rmsnorm(&desc.raw) }
    }

    pub(crate) unsafe fn rope_apply_bf16(&mut self, desc: &RopeApplyBf16) -> Result<(), Status> {
        unsafe { self.context.rope_apply(&desc.raw) }
    }

    pub(crate) unsafe fn gdn_decode_bf16(&mut self, desc: &GdnDecodeBf16) -> Result<(), Status> {
        unsafe { crate::ffi::qscu::gdn_decode(self.context, &desc.raw) }
    }

    pub(crate) unsafe fn gdn_prefill_bf16(&mut self, desc: &GdnPrefillBf16) -> Result<(), Status> {
        unsafe { crate::ffi::qscu::gdn_prefill(self.context, &desc.raw) }
    }

    pub(crate) unsafe fn create_moe_bf16_plan(
        &mut self,
        config: MoeBf16PlanConfig,
    ) -> Result<MoePlan, Status> {
        let desc = config.desc()?;
        unsafe { self.context.create_moe_plan(&desc) }
    }

    pub(crate) unsafe fn moe_workspace_size(
        &mut self,
        plan: &MoePlan,
        num_tokens: u32,
    ) -> Result<usize, Status> {
        unsafe { self.context.moe_workspace_size(plan, num_tokens) }
    }

    pub(crate) unsafe fn moe_execute_bf16(
        &mut self,
        plan: &MoePlan,
        desc: &MoeBf16Execute,
    ) -> Result<(), Status> {
        unsafe { self.context.moe_execute_bf16(plan, &desc.raw) }
    }
}
