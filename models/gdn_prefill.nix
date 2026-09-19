# AOT stages of Qwen GDN prefill, in execution order.
# Fixed chunk size and arithmetic live in triton_kernels/gdn_prefill_*.py.
{ text, precision }:
let
  keyHeads = text.linear_num_key_heads;
  valueHeads = text.linear_num_value_heads;
  keyDim = text.linear_key_head_dim;
  valueDim = text.linear_value_head_dim;
in
assert keyHeads == 16 && (valueHeads == 32 || valueHeads == 48) && keyDim == 128 && valueDim == 128;
{
  # Split convolution output, normalize Q/K, and compute decay and beta.
  gdn_prefill_prep = {
    provider = "triton";
    source = "triton_kernels/gdn_prefill_prep.py";
    spec = {
      grid = null; # Rust supplies dimensions from the current token count.
      precision = {
        mixed_qkv_ptr = "bf16";
        a_ptr = "bf16";
        b_ptr = "bf16";
        A_log_ptr = "bf16";
        dt_bias_ptr = "bf16";
        q_ptr = "bf16";
        k_ptr = "bf16";
        v_ptr = "bf16";
        g_ptr = "f32";
        beta_ptr = "f32";
      };
      constants = {
        H = keyHeads;
        HV = valueHeads;
        K = keyDim;
        V = valueDim;
      };
      options = {
        num_stages = 2;
        num_warps = 4;
      };
    };
  };

  # Accumulate log decay within each 64-token chunk.
  gdn_prefill_cumsum = {
    provider = "triton";
    source = "triton_kernels/gdn_prefill_cumsum.py";
    spec = {
      grid = null; # Rust supplies dimensions from the current token count.
      precision = {
        s = "f32";
        o = "f32";
        cu_seqlens = "i32";
        chunk_indices = "i32";
      };
      constants = {
        H = valueHeads;
      };
      options = {
        num_stages = 1;
        num_warps = 4;
      };
    };
  };

  # Build the decay-weighted, beta-scaled key interaction matrix.
  gdn_prefill_kkt = {
    provider = "triton";
    source = "triton_kernels/gdn_prefill_kkt.py";
    spec = {
      grid = null; # Rust supplies dimensions from the current token count.
      precision = {
        k = "bf16";
        beta = "f32";
        g = "f32";
        A = "f32";
        cu_seqlens = "i32";
        chunk_indices = "i32";
      };
      constants = {
        H = valueHeads;
        Hg = keyHeads;
        K = keyDim;
      };
      options = {
        num_stages = 2;
        num_warps = 4;
      };
    };
  };

  # Invert its unit lower-triangular system.
  gdn_prefill_solve = {
    provider = "triton";
    source = "triton_kernels/gdn_prefill_solve.py";
    spec = {
      grid = null; # Rust supplies dimensions from the current token count.
      precision = {
        A = "f32";
        Ai = "bf16";
        cu_seqlens = "i32";
        chunk_indices = "i32";
      };
      constants = {
        H = valueHeads;
      };
      options = {
        num_stages = 2;
        num_warps = 4;
      };
    };
  };

  # Apply the inverse to form transformed keys and values.
  gdn_prefill_wu = {
    provider = "triton";
    source = "triton_kernels/gdn_prefill_wu.py";
    spec = {
      grid = null; # Rust supplies dimensions from the current token count.
      precision = {
        k = "bf16";
        v = "bf16";
        beta = "f32";
        w = "bf16";
        u = "bf16";
        A = "bf16";
        g = "f32";
        cu_seqlens = "i32";
        chunk_indices = "i32";
      };
      constants = {
        H = valueHeads;
        Hg = keyHeads;
        K = keyDim;
        V = valueDim;
      };
      options = {
        num_stages = 2;
        num_warps = 4;
      };
    };
  };

  # Advance recurrent state across chunks and save chunk input states.
  gdn_prefill_state = {
    provider = "triton";
    source = "triton_kernels/gdn_prefill_state.py";
    spec = {
      grid = null; # Rust supplies dimensions from the current token count.
      precision = {
        k = "bf16";
        v = "bf16";
        w = "bf16";
        v_new = "bf16";
        g = "f32";
        gk = "f32";
        h = "bf16";
        h0 = precision.gdn.recurrentState;
        ht = precision.gdn.recurrentState;
        cu_seqlens = "i32";
        chunk_offsets = "i32";
      };
      constants = {
        H = valueHeads;
        Hg = keyHeads;
        K = keyDim;
        V = valueDim;
      };
      options = {
        num_stages = 2;
        num_warps = 4;
      };
    };
  };

  # Combine chunk input states with within-chunk attention outputs.
  gdn_prefill_output = {
    provider = "triton";
    source = "triton_kernels/gdn_prefill_output.py";
    spec = {
      grid = null; # Rust supplies dimensions from the current token count.
      precision = {
        q = "bf16";
        k = "bf16";
        v = "bf16";
        h = "bf16";
        g = "f32";
        o = "bf16";
        cu_seqlens = "i32";
        chunk_indices = "i32";
      };
      constants = {
        H = valueHeads;
        Hg = keyHeads;
        K = keyDim;
        V = valueDim;
      };
      options = {
        num_stages = 2;
        num_warps = 4;
      };
    };
  };

}
