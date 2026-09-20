# Engine build configuration; Triton LM-head compilation consumes this today.
# nix eval --json --file models/config.nix
#
# Select one model and target per engine. This is a place to record facts and
# implementation decisions for future C/Rust/Python generators, not a runtime
# options API. build_tools/ will hold the generators and compiler adapters.
let
  mod = a: b: a - (builtins.div a b) * b;
  nextPowerOfTwo = n: x: if x >= n then x else nextPowerOfTwo n (x * 2);
  readModel = directory: {
    source = import (directory + "/source.nix");
    config = (builtins.fromJSON (builtins.readFile (directory + "/model.json"))).text_config;
  };
in
let
  engine = {
    model = "qwen3.8-27b-nvfp4";
    mtp = false;
  };

  # Keep upstream JSON untouched. Source revisions document our build inputs;
  # they do not restrict which finetune's weights the resulting engine can load.
  models = {
    "qwen3.6-35b-a3b" = readModel ./qwen3.6-35b-a3b;
    "qwen3.6-27b" = readModel ./qwen3.6-27b;
    "qwen3.8-27b" = readModel ./qwen3.8-27b;
    "qwen3.6-35b-a3b-nvfp4" = readModel ./qwen3.6-35b-a3b-nvfp4;
    "qwen3.6-27b-nvfp4" = readModel ./qwen3.6-27b-nvfp4;
    "qwen3.8-27b-nvfp4" = readModel ./qwen3.8-27b-nvfp4;
  };
  model = models.${engine.model};
  text = model.config;

  # This runtime targets SM121. Keep its hardware facts together; no host
  # detection or target selection. Warp size is required by the AOT compiler.
  target = {
    backend = "cuda";
    computeCapability = {
      major = 12;
      minor = 1;
    };
    warpSize = 32;
  };

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
    gdn = rec {
      packedQkvChannels =
        2 * text.linear_num_key_heads * text.linear_key_head_dim
        + text.linear_num_value_heads * text.linear_value_head_dim;
      outputWidth = text.linear_num_value_heads * text.linear_value_head_dim;
      convWidth = text.linear_conv_kernel_dim;
      convHistoryLen = convWidth - 1;
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
    let
      triton = name: blocks: spec: {
        provider = "triton";
        source = "triton_kernels/${name}.py";
        spec = {
          grid = [
            blocks
            1
            1
          ];
          options.num_warps = 4;
        }
        // spec;
      };
      samplingBlocks = builtins.div (text.vocab_size + 1023) 1024;
      gemv =
        rows: overrides:
        triton "gemv" rows (
          {
            precision = precision.projections;
            constants = {
              K = text.hidden_size;
              BLOCK_K = nextPowerOfTwo text.hidden_size 1;
              ACC.dtype = precision.projections.accumulation;
            };
            options = {
              num_warps = 8;
              num_stages = 1;
              enable_fp_fusion = false;
            };
          }
          // overrides
        );
    in
    {
      # Sketch of existing provider choices, not a claim of completed 27B support.
      projections = {
        provider = "cublaslt";
      };
      nvfp4 = {
        provider = "cutlass";
        # GB10 measurements: keep narrow decode tiles; widen for prefill.
        # QuTLASS's SM120 256x128x128 recipe wins at 512/1024 rows.
        mediumRows = 128;
        mediumTactic = "tile128x64_dp";
        prefillRows = 512;
        prefillTactic = "qutlass256x128";
        smallTactic = "tile128x32_dp";
        # Available dense W4A4 tactics are tileN × streamK.
        # Preparation chooses one; tests exercise every enabled combination.
        # Each tile compiles both schedulers; streamK controls API availability.
        tileN = [
          32
          64
        ];
        streamK = [
          false
          true
        ];
      };
      lm_head = gemv text.vocab_size { precision = precision.lm_head; };
      gdn_qkv = gemv dimensions.gdn.packedQkvChannels { };
      # KR03: only M=1/N=10240/K=5120 is qualified for this FP8 replacement.
      fp8_decode = {
        provider = "cute";
        source = "cute_kernels/fp8_decode.py";
        spec = {
          precision = {
            x = "fp8_e4m3";
            w = "fp8_e4m3";
            partials = "f32";
            input_scale = "f32";
            weight_scale = "f32";
          };
          alignments = {
            x = 16;
            w = 16;
            partials = 16;
            input_scale = 4;
            weight_scale = 4;
          };
          constants = {
            N = 10240;
            K = 5120;
          };
          options = {
            "gpu-arch" = "sm_121a";
            "host-target" = "linux-aarch64";
          };
        };
      };
      nvfp4_up = {
        provider = "cute";
        source = "cute_kernels/nvfp4_gemm.py";
        spec = {
          precision = { a = "nvfp4_e2m1"; b = "nvfp4_e2m1"; sfa = "fp8_e4m3"; sfb = "fp8_e4m3"; output = "bf16"; alpha = "f32"; };
          alignments = { a = 16; b = 16; sfa = 16; sfb = 16; output = 16; alpha = 4; };
          constants = { N = 17408; K = 5120; };
          options = { "gpu-arch" = "sm_121a"; "host-target" = "linux-aarch64"; };
        };
      };
      nvfp4_down = {
        provider = "cute";
        source = "cute_kernels/nvfp4_gemm.py";
        spec = {
          precision = { a = "nvfp4_e2m1"; b = "nvfp4_e2m1"; sfa = "fp8_e4m3"; sfb = "fp8_e4m3"; output = "bf16"; alpha = "f32"; };
          alignments = { a = 16; b = 16; sfa = 16; sfb = 16; output = 16; alpha = 4; };
          constants = { N = 5120; K = 17408; };
          options = { "gpu-arch" = "sm_121a"; "host-target" = "linux-aarch64"; };
        };
      };
      nvfp4_head = {
        provider = "cute";
        source = "cute_kernels/nvfp4_gemm.py";
        spec = {
          precision = { a = "nvfp4_e2m1"; b = "nvfp4_e2m1"; sfa = "fp8_e4m3"; sfb = "fp8_e4m3"; output = "bf16"; alpha = "f32"; };
          alignments = { a = 16; b = 16; sfa = 16; sfb = 16; output = 16; alpha = 4; };
          constants = { N = 248320; K = 5120; };
          options = { "gpu-arch" = "sm_121a"; "host-target" = "linux-aarch64"; };
        };
      };
      fp8_reduce = triton "fp8_reduce" 10 {
        precision = {
          partials = "f32";
          output = "bf16";
        };
        constants = {
          N = 10240;
          BLOCK = 1024;
        };
      };
      sampling_prepare = triton "sampling_prepare" samplingBlocks {
        precision = {
          logits = "f32";
          output = "f32";
        };
        constants = {
          VOCAB = text.vocab_size;
          BLOCK = 1024;
        };
      };
      sampling_filter = triton "sampling_filter" 1 {
        precision = {
          LOGITS = "f32";
          BUFFER = "f32";
          PERCENTILE_TO_STD_TABLE = "f32";
          NORMAL_CDF_TO_SIGMA_TABLE = "f32";
        };
        constants = {
          BATCH_SIZE = 1;
          VOCAB_SIZE = text.vocab_size;
          BLOCK_SIZE = 8192;
          BLOCK_SIZE_TRUNC = 4096;
        };
      };
      sampling_gumbel = triton "sampling_gumbel" samplingBlocks {
        precision = {
          logits = "f32";
          original = "f32";
          local_max = "f32";
          local_ids = "i32";
          position = "i32";
        };
        constants.BLOCK = 1024;
      };
      sampling_reduce = triton "sampling_reduce" 1 {
        precision = {
          local_max = "f32";
          local_ids = "i32";
          output = "i32";
        };
        constants = {
          BLOCKS = samplingBlocks;
          BLOCK = nextPowerOfTwo samplingBlocks 1;
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
        # Keep the MoE choice defined for both Rust paths to type-check. Dense
        # execution ignores it; hasExperts selects the active implementation.
        {
          bf16Kernel = "decode_gemv64";
        }
        // (
          if hasExperts then
            {
              provider = "flashinfer-cutlass";
              router = {
                score = "softmax";
                renormalize = true;
                scalingFactor = 1.0;
              };
            }
          else
            { provider = "cublaslt"; }
        );

    }
    // (import ./gdn_prefill.nix { inherit text precision; });

  # Quantized recipes will also need concrete packing and scale layouts.
  # Compiler dependencies stay pinned in flake.lock and build_tools/uv.lock.
  # Batch/sequence sizes, memory budgets and request state remain runtime data.
  # Loading checks architecture/layout compatibility and safe tensor addressing,
  # not weight hashes, source revisions, or byte-identical config JSON.
}
