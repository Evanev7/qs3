// Standalone kernel experiment. C++ owns benchmark fixtures only; production
// scheduling/memory remain in Rust. No Python, PTX JIT or autotuning at runtime.
#include "qscb.h"
#include "kernels.h"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
static void driver_check(CUresult e) {
    if (e != CUDA_SUCCESS) {
        const char* s = nullptr; cuGetErrorString(e, &s);
        std::fprintf(stderr, "CUDA driver: %s\n", s ? s : "unknown error"); std::exit(1);
    }
}
static void linear_check(qsfi_status e, qscb_context* ctx) {
    if (e != QSFI_STATUS_OK) {
        qsfi_error_info info{}; qscb_context_get_last_error(ctx, &info);
        std::fprintf(stderr, "qscb: %s\n", info.message); std::exit(1);
    }
}
template<class T> struct Buffer {
    T* ptr{};
    explicit Buffer(size_t n) { cuda_check(cudaMalloc(&ptr, n * sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
static qsfi_tensor2 matrix(void* p, qsfi_dtype dtype, int n, int k) {
    qsfi_tensor2 t{}; t.data=p; t.dtype=dtype;
    t.shape[0]=n; t.shape[1]=k; t.stride[0]=k; t.stride[1]=1; return t;
}
// Reproducible, nonzero BF16 inputs with signs and cancellation. CPU reference
// reconstructs selected rows independently of the GPU initialization kernel.
__host__ __device__ static float value(size_t i, unsigned seed) {
    unsigned h = unsigned(i) ^ seed;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return __bfloat162float(__float2bfloat16(float(int(h % 1009) - 504) * 0.00071f));
}
__global__ static void fill(__nv_bfloat16* p, size_t n, unsigned seed) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<n;
         i+=size_t(blockDim.x)*gridDim.x) p[i]=__float2bfloat16(value(i, seed));
}
__global__ static void evict(unsigned* p, size_t n) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<n;
         i+=size_t(blockDim.x)*gridDim.x) p[i] += 1;
}
static std::vector<float> download(void* p, int n, bool fp32, cudaStream_t stream) {
    std::vector<float> result(n);
    if (fp32) {
        cuda_check(cudaMemcpyAsync(result.data(), p, n*4ull, cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
    } else {
        std::vector<__nv_bfloat16> raw(n);
        cuda_check(cudaMemcpyAsync(raw.data(), p, n*2ull, cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        for(int i=0;i<n;++i) result[i]=__bfloat162float(raw[i]);
    }
    return result;
}
static void validate(const KernelSpec& s, const std::vector<float>& ref,
                     const std::vector<float>& actual) {
    float max_abs=0;
    // cuBLASLt comparison covers every output, CPU double covers 65 rows,
    // including both boundaries. Tolerances are probe gates, not model gates.
    const float atol=s.fp32 ? 0.0002f : 0.01f;
    const float rtol=s.fp32 ? 0.0002f : 0.01f;
    for(int i=0;i<s.n;++i) {
        const float diff=std::fabs(actual[i]-ref[i]); max_abs=std::max(max_abs,diff);
        if (!std::isfinite(actual[i]) || !std::isfinite(ref[i]) || diff>atol+rtol*std::fabs(ref[i])) {
            std::fprintf(stderr,"cuBLAS mismatch %s row=%d got=%g ref=%g\n",s.name,i,actual[i],ref[i]);
            std::exit(1);
        }
    }
    for(int sample=0;sample<=64;++sample) {
        int row=int(size_t(sample)*(s.n-1)/64); double expected=0;
        for(int col=0;col<s.k;++col)
            expected+=double(value(col,3))*double(value(size_t(row)*s.k+col,7));
        if (!s.fp32) expected=__bfloat162float(__float2bfloat16(float(expected)));
        for(float got : {ref[row],actual[row]}) if(std::fabs(got-expected)>atol+rtol*std::fabs(expected)) {
            std::fprintf(stderr,"CPU mismatch %s row=%d got=%g ref=%.12g\n",s.name,row,got,expected);
            std::exit(1);
        }
    }
    std::fprintf(stderr,"validated %s rows=%d warps=%d max_abs=%g\n",s.name,s.rows,s.warps,max_abs);
}
template<class F> static float measure(F launch, cudaStream_t stream, unsigned* eviction,
                                      size_t eviction_words, bool cold) {
    cudaEvent_t start,stop;
    cuda_check(cudaEventCreate(&start)); cuda_check(cudaEventCreate(&stop));
    for(int i=0;i<10;++i) launch();
    cuda_check(cudaStreamSynchronize(stream));
    std::vector<float> samples;
    for(int i=0;i<50;++i) {
        if(cold) { evict<<<4096,256,0,stream>>>(eviction,eviction_words); cuda_check(cudaGetLastError()); }
        cuda_check(cudaEventRecord(start,stream)); launch();
        cuda_check(cudaEventRecord(stop,stream)); cuda_check(cudaEventSynchronize(stop));
        float ms; cuda_check(cudaEventElapsedTime(&ms,start,stop)); samples.push_back(ms*1000);
    }
    cuda_check(cudaEventDestroy(start)); cuda_check(cudaEventDestroy(stop));
    std::sort(samples.begin(),samples.end()); return (samples[24]+samples[25])*0.5f;
}
int main(int argc, char** argv) {
    if(argc!=2) { std::fprintf(stderr,"usage: probe ARTIFACT_DIRECTORY\n"); return 2; }
    cudaDeviceProp prop{}; cuda_check(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=12 || prop.minor!=1) { std::fprintf(stderr,"requires SM121\n"); return 2; }
    int driver,runtime; cuda_check(cudaDriverGetVersion(&driver)); cuda_check(cudaRuntimeGetVersion(&runtime));
    std::fprintf(stderr,"device=%s driver=%d runtime=%d weights=cudaMalloc eager=true\n",prop.name,driver,runtime);
    cudaStream_t stream; cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    qscb_context* ctx{}; qscb_context_desc context_desc{0,stream};
    linear_check(qscb_context_create(&context_desc,&ctx),ctx);
    {
        Buffer<unsigned char> workspace(64ull<<20);
        const size_t eviction_words=std::max(size_t(128ull<<20),size_t(prop.l2CacheSize)*4)/sizeof(unsigned);
        Buffer<unsigned> eviction(eviction_words);
        cuda_check(cudaMemsetAsync(eviction.ptr,0,eviction_words*4,stream));
        std::puts("shape,n,k,dtype,rows,warps,cache,repeat,cublas_p50_us,triton_p50_us");
        for(const auto& s:kernels) {
            Buffer<__nv_bfloat16> x(s.k),weight(size_t(s.n)*s.k);
            Buffer<unsigned char> ref(size_t(s.n)*(s.fp32?4:2)),out(size_t(s.n)*(s.fp32?4:2));
            fill<<<4096,256,0,stream>>>(x.ptr,s.k,3);
            fill<<<4096,256,0,stream>>>(weight.ptr,size_t(s.n)*s.k,7); cuda_check(cudaGetLastError());
            qscb_linear_desc d{};
            d.x=matrix(x.ptr,QSFI_DTYPE_BF16,1,s.k);
            d.weight=matrix(weight.ptr,QSFI_DTYPE_BF16,s.n,s.k);
            d.out=matrix(ref.ptr,s.fp32?QSFI_DTYPE_F32:QSFI_DTYPE_BF16,1,s.n);
            d.rows=1; d.in_features=s.k; d.out_features=s.n; d.alpha=1;
            d.workspace=workspace.ptr; d.workspace_bytes=64ull<<20;
            qscb_linear_plan* plan{}; linear_check(qscb_linear_plan_create(ctx,&d,&plan),ctx);
            CUmodule module; CUfunction function;
            driver_check(cuModuleLoad(&module,(std::string(argv[1])+"/"+s.file).c_str()));
            driver_check(cuModuleGetFunction(&function,module,s.symbol));
            if(s.shared) driver_check(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,s.shared));
            auto baseline=[&]{linear_check(qscb_linear_execute(ctx,plan,&d),ctx);};
            auto candidate=[&]{
                // Triton's generated NVIDIA ABI reserves two scratch pointers.
                // The exporter rejects any nonzero requirement for either.
                CUdeviceptr scratch=0,profile=0;
                void* params[]={&x.ptr,&weight.ptr,&out.ptr,&scratch,&profile};
                driver_check(cuLaunchKernel(function,(s.n+s.rows-1)/s.rows,1,1,
                                            s.warps*32,1,1,s.shared,stream,params,nullptr));
            };
            baseline(); candidate();
            validate(s,download(ref.ptr,s.n,s.fp32,stream),download(out.ptr,s.n,s.fp32,stream));
            for(bool cold:{false,true}) for(int repeat=0;repeat<3;++repeat) {
                float a,b;
                if(repeat%2==0) {
                    a=measure(baseline,stream,eviction.ptr,eviction_words,cold);
                    b=measure(candidate,stream,eviction.ptr,eviction_words,cold);
                } else {
                    b=measure(candidate,stream,eviction.ptr,eviction_words,cold);
                    a=measure(baseline,stream,eviction.ptr,eviction_words,cold);
                }
                std::printf("%s,%d,%d,%s,%d,%d,%s,%d,%.3f,%.3f\n",s.name,s.n,s.k,
                            s.fp32?"fp32":"bf16",s.rows,s.warps,cold?"evicted":"repeated",repeat,a,b);
                std::fflush(stdout);
            }
            cuda_check(cudaStreamSynchronize(stream));
            driver_check(cuModuleUnload(module)); qscb_linear_plan_destroy(plan);
        }
    }
    qscb_context_destroy(ctx); cuda_check(cudaStreamDestroy(stream)); return 0;
}
