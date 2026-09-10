# Engine build configuration; Triton LM-head compilation consumes this today.
# nix eval --json --file models/config.nix
#
# Select one model and target per engine. This is a place to record facts and
# implementation decisions for future C/Rust/Python generators, not a runtime
# options API. build_tools/ will hold the generators and compiler adapters.
let
  mod = a: b: a - (builtins.div a b) * b;
  nextPowerOfTwo =
    n:
    let
      go = x: if x >= n then x else go (x * 2);
    in
    go 1;
  readModel = directory: {
    source = import (directory + "/source.nix");
    config = (builtins.fromJSON (builtins.readFile (directory + "/model.json"))).text_config;
  };
in
let
  engine = {
    model = "qwen3.6-35b-a3b";
    target = "sm121";
    mtp = false;
  };

  # Keep upstream JSON untouched. Source revisions document our build inputs;
  # they do not restrict which finetune's weights the resulting engine can load.
  models = {
    "qwen3.6-35b-a3b" = readModel ./qwen3.6-35b-a3b;
    "qwen3.6-27b" = readModel ./qwen3.6-27b;
  };
  model = models.${engine.model};
  text = model.config;

  targets = {
    sm121 = {
      backend = "cuda";
      computeCapability = {
        major = 12;
        minor = 1;
      };
      warpSize = 32; # Hardware fact, not a kernel tuning parameter.
    };
    # Backend placeholder. Add concrete Apple GPU requirements and kernel
    # choices when implementing Metal; do not inherit CUDA hardware facts.
    metal = {
      backend = "metal";
    };
  };
  target = targets.${engine.target};

  # Normalize upstream dtype names once. Language-specific type spellings and
  # formatted compiler target strings belong in the respective build adapters.
  dtypes = {
    bfloat16 = "bf16";
    float32 = "f32";
  };
  bf16Linear = {
    weight = dtypes.${text.dtype};
    activation = "bf16";
    accumulation = "f32";
    output = "bf16";
  };
  hasExperts = text ? num_experts;
  precision = {
    embedding = {
      weight = dtypes.${text.dtype};
      output = "bf16";
    };
    residual = "bf16";
    norm = bf16Linear;
    projections = bf16Linear;
    lm_head = bf16Linear // {
      output = "f32";
    };
    kvCache = "bf16";
    gdn = {
      convState = "bf16";
      recurrentState = dtypes.${text.mamba_ssm_dtype};
      recurrenceAccumulation = "f32";
      output = "bf16";
    };
    mlp =
      if hasExperts then
        {
          routedExperts = bf16Linear;
          sharedExpert = bf16Linear;
          router = bf16Linear // {
            output = "f32";
            routeWeight = "f32";
            routeIndex = "i32";
          };
        }
      else
        { dense = bf16Linear; };
  };

  # Derived dimensions used by kernel specialization and layout generation.
  # Layer ordering/count and other model facts remain in model.config; there is
  # no second handwritten layer schedule or model-shape table here.
  dimensions = {
    attention = rec {
      headDim = text.head_dim;
      groupSize = builtins.div text.num_attention_heads text.num_key_value_heads;
      qWidth = text.num_attention_heads * headDim;
      kvWidth = text.num_key_value_heads * headDim;
      packedQGateWidth = qWidth * (if text.attn_output_gate then 2 else 1);
      rotaryDim = builtins.floor (headDim * text.rope_parameters.partial_rotary_factor);
    };
    gdn = {
      packedWidth =
        2 * text.linear_num_key_heads * text.linear_key_head_dim
        + text.linear_num_value_heads * text.linear_value_head_dim;
      outputWidth = text.linear_num_value_heads * text.linear_value_head_dim;
      convHistory = text.linear_conv_kernel_dim - 1;
    };
  };

in
assert text.num_key_value_heads > 0;
assert mod text.num_attention_heads text.num_key_value_heads == 0;
assert
  let
    rotary = dimensions.attention.rotaryDim;
  in
  rotary > 0
  && rotary <= text.head_dim
  && rotary == builtins.floor rotary
  && mod (builtins.floor rotary) 2 == 0;
{
  inherit
    engine
    model
    target
    precision
    dimensions
    ;

  # Enabled MTP becomes part of this engine build; no runtime model dispatch.
  # Kernel recipes and verification scheduling are still to be designed.
  mtp = {
    layers = if engine.mtp then text.mtp_num_hidden_layers else 0;
    dedicatedEmbeddings = engine.mtp && text.mtp_use_dedicated_embeddings;
  };

  kernels =
    if target.backend == "cuda" then
      {
        # Sketch of existing provider choices, not a claim of completed 27B support.
        projections = {
          provider = "cublaslt";
        };
        lm_head = {
          provider = "triton";
          precision = precision.lm_head;
          constants = {
            K = text.hidden_size;
            BLOCK_K = nextPowerOfTwo text.hidden_size;
            ACC = {
              dtype = precision.lm_head.accumulation;
            };
          };
          grid = [
            text.vocab_size
            1
            1
          ];
          options = {
            num_warps = 8;
            num_stages = 1;
            enable_fp_fusion = false;
          };
        };
        attention = {
          provider = "flashinfer";
          pdl = true;
          posEncoding = "none"; # Rust schedules explicit q/k norm + partial RoPE.
          ctaTileQ = [
            16
            32
            64
            128
          ];
          groupSize = dimensions.attention.groupSize;
          headDim = dimensions.attention.headDim;
        };
        gdn = {
          provider = "qscu";
          prepThreads = 128;
          softplusBeta = 1.0;
          softplusThreshold = 20.0;
        };
        mlp =
          if hasExperts then
            {
              provider = "flashinfer-cutlass";
              bf16Kernel = "tile32-blocks96";
              router = {
                score = "softmax";
                renormalize = true;
                scalingFactor = 1.0;
              };
            }
          else
            { provider = "cublaslt"; };

      }
    else
      null; # Metal implementation recipes still to come.

  # Quantized recipes will also need concrete packing and scale layouts.
  # Compiler dependencies stay pinned in flake.lock and build_tools/uv.lock.
  # Batch/sequence sizes, memory budgets and request state remain runtime data.
  # Loading checks architecture/layout compatibility and safe tensor addressing,
  # not weight hashes, source revisions, or byte-identical config JSON.
}
