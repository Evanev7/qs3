# Numerical reference inputs only. Test sizes and inputs belong to the generators.
let
  config = import ../../models/config.nix;
  text = config.model.config;
  attention = config.dimensions.attention;
in
builtins.toJSON {
  hidden_size = text.hidden_size;
  q_heads = text.num_attention_heads;
  kv_heads = text.num_key_value_heads;
  head_dim = text.head_dim;
  rotary_dim = attention.rotaryDim;
  q_hidden = attention.qWidth;
  kv_hidden = attention.kvWidth;
  q_proj_out = attention.packedQGateWidth;
  group_size = attention.groupSize;
  rms_eps = text.rms_norm_eps;
  rope_theta = text.rope_parameters.rope_theta;
  key_heads = text.linear_num_key_heads;
  value_heads = text.linear_num_value_heads;
  key_dim = text.linear_key_head_dim;
  value_dim = text.linear_value_head_dim;
  conv_width = text.linear_conv_kernel_dim;
  qkv_dim = config.dimensions.gdn.packedQkvChannels;
  gdn_output_dim = config.dimensions.gdn.outputWidth;
  has_experts = text ? num_experts;
}
