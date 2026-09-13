#include "qscu.cu"
#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstdlib>

__global__ void control_argmax(greedy_argmax_params p)
{
    const uint32_t row = blockIdx.x;
    float best_score = kRouterNegInf;
    uint32_t best_token = UINT_MAX;

    for (uint32_t token = threadIdx.x; token < p.vocab_size; token += blockDim.x) {
        float score = p.logits
                          [static_cast<int64_t>(row) * p.logits_stride0
                           + static_cast<int64_t>(token) * p.logits_stride1];
        if (better_argmax_candidate(score, token, best_score, best_token)) {
            best_score = score;
            best_token = token;
        }
    }

    __shared__ float scores[256];
    __shared__ uint32_t tokens[256];
    scores[threadIdx.x] = best_score;
    tokens[threadIdx.x] = best_token;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride
            && better_argmax_candidate(
                scores[threadIdx.x + stride],
                tokens[threadIdx.x + stride],
                scores[threadIdx.x],
                tokens[threadIdx.x]
            )) {
            scores[threadIdx.x] = scores[threadIdx.x + stride];
            tokens[threadIdx.x] = tokens[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (p.out_i32 != nullptr)
            p.out_i32[static_cast<int64_t>(row) * p.out_stride0] = static_cast<int32_t>(tokens[0]);
        else
            p.out_u32[static_cast<int64_t>(row) * p.out_stride0] = tokens[0];
    }
}

template<int Threads,int Vec>
__global__ void candidate_argmax(greedy_argmax_params p) {
    float best=kRouterNegInf; uint32_t token=UINT_MAX;
    const float* logits=p.logits+int64_t(blockIdx.x)*p.logits_stride0;
    for(uint32_t base=threadIdx.x*Vec;base<p.vocab_size;base+=Threads*Vec) {
        #pragma unroll
        for(int j=0;j<Vec;++j) {
            uint32_t t=base+j;
            if(t<p.vocab_size) {
                float v=logits[int64_t(t)*p.logits_stride1];
                if(better_argmax_candidate(v,t,best,token)){best=v;token=t;}
            }
        }
    }
    for(int d=16;d;d/=2) {
        float v=__shfl_down_sync(0xffffffff,best,d);
        uint32_t t=__shfl_down_sync(0xffffffff,token,d);
        if(better_argmax_candidate(v,t,best,token)){best=v;token=t;}
    }
    __shared__ float scores[Threads/32];
    __shared__ uint32_t ids[Threads/32];
    if(threadIdx.x%32==0){scores[threadIdx.x/32]=best;ids[threadIdx.x/32]=token;}
    __syncthreads();
    if(threadIdx.x<32){
        best=threadIdx.x<Threads/32?scores[threadIdx.x]:kRouterNegInf;
        token=threadIdx.x<Threads/32?ids[threadIdx.x]:UINT_MAX;
        for(int d=16;d;d/=2) {
            float v=__shfl_down_sync(0xffffffff,best,d);
            uint32_t t=__shfl_down_sync(0xffffffff,token,d);
            if(better_argmax_candidate(v,t,best,token)){best=v;token=t;}
        }
        if(threadIdx.x==0)p.out_i32[int64_t(blockIdx.x)*p.out_stride0]=int32_t(token);
    }
}
void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(){
    constexpr int N=248320;
    std::vector<float> h(N);for(int i=0;i<N;++i)h[i]=float((i*17)%1009);
    h[240000]=2000;h[200001]=2000;
    float* x;int32_t* y;check(cudaMalloc(&x,N*4));check(cudaMalloc(&y,4));check(cudaMemcpy(x,h.data(),N*4,cudaMemcpyHostToDevice));
    greedy_argmax_params p{};p.logits=x;p.logits_stride0=N;p.logits_stride1=1;p.out_i32=y;p.out_stride0=1;p.vocab_size=N;
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));
    for(int rep=0;rep<3;++rep)for(int v=0;v<7;++v){
        auto launch=[&](){switch(v){
            case 0:control_argmax<<<1,256>>>(p);break;
            case 1:candidate_argmax<256,1><<<1,256>>>(p);break;
            case 2:candidate_argmax<512,1><<<1,512>>>(p);break;
            case 3:candidate_argmax<1024,1><<<1,1024>>>(p);break;
            case 4:candidate_argmax<256,4><<<1,256>>>(p);break;
            case 5:candidate_argmax<512,4><<<1,512>>>(p);break;
            case 6:candidate_argmax<1024,4><<<1,1024>>>(p);break;
        }};
        launch();check(cudaGetLastError());int32_t got;check(cudaMemcpy(&got,y,4,cudaMemcpyDeviceToHost));if(got!=200001)return 2;
        check(cudaEventRecord(a));for(int i=0;i<200;++i)launch();check(cudaEventRecord(b));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));printf("repeat=%d variant=%d us=%.3f id=%d\n",rep,v,ms*5,got);
    }
    check(cudaFree(x));check(cudaFree(y));check(cudaEventDestroy(a));check(cudaEventDestroy(b));
}
