#include <cuda.h>
#include <stdio.h>
#include <stdint.h> 


// For comparisons
//#include "seqScan.c"


#define N 64*64


/* ------------------------------------------------------------------------
   Unrolled in-place(shared memory) Scan without syncs (32 threads, 64 elts). 
   Needs 2*64 elements of shared memory storage (512bytes).
     (shared mem is 49152 bytes, but you share it with other blocks on an MP) 
   --------------------------------------------------------------------- */
__device__ void skl_scan(int i, 
			 float* input, 
			 float *output, 
			 float *s_data, // The shared memory
			 float *maxs) { 
    
  int tid = threadIdx.x; 
  int tids = tid << 1; 

  int eltOffs = blockIdx.x * 64 + tid; 

  // Load data from global memory into shared memory (in two separate load ops)
  s_data[tid] = input[eltOffs]; 
  s_data[tid + 32] = input[eltOffs + 32]; 
  // NO SYNC HERE 

  s_data[tids | 1] += s_data[tids]; 
  s_data[(tids | 3) - (tid & 1)] += s_data[tids & 0xFFFFFFFC | 1]; 
  s_data[(tids | 7) - (tid & 3)] += s_data[tids & 0xFFFFFFF8 | 3]; 
  s_data[(tids | 15) - (tid & 7)] += s_data[tids & 0xFFFFFFF0 | 7]; 
  s_data[(tids | 31) - (tid & 15)] += s_data[tids & 0xFFFFFFE0 | 15]; 
  s_data[(tids | 63) - (tid & 31)] += s_data[tids & 0xFFFFFFC0 | 31];
  // NO Interleaved SYNCS here.

  output[eltOffs] = s_data[tid]; 
  output[eltOffs + 32] = s_data[tid + 32];
  
  if(tid == 0) 
    maxs[i] = s_data[63];
  
}


/* ------------------------------------------------------------------------
   The Scan kernel (Thousand(s) of elements. NO SYNCS AT ALL) 
   --------------------------------------------------------------------- */
__global__ void kernel(float* input0,
                       float* output0, 
                       float* maxout){
   
  // shared data. (two different kinds. warp local and across warps.) 
  extern __shared__ float s_data[]; 
  float *maxs = &s_data[64]; 
  
  // Sequentially execute 64 scans
  for (int i = 0; i < 64; i ++) { 
    skl_scan(i,input0+i*64,output0+i*64,s_data,maxs);
  }
  
  // in parallel scan the maximum array 
  float v; //discard this value.
  skl_scan(0,maxs,maxs,(float *)s_data,&v);
  

  // distribute (now in two phases) 

  // 31 thread pass. 
  if (threadIdx.x > 0) {
    for (int j = 0; j < 64; j ++) {
      output0[(blockIdx.x*64)+(threadIdx.x*64+j)] += maxs[threadIdx.x-1];
    }
  }

  // 32 thread pass. 
  for (int j = 0; j < 64; j ++) {
    output0[((blockIdx.x+32)*64)+(threadIdx.x*64+j)] += maxs[threadIdx.x+31];
  }

  // This is a debug step. 
  maxout[threadIdx.x] = maxs[threadIdx.x];
  maxout[threadIdx.x+32] = maxs[threadIdx.x+32];

}

/* ------------------------------------------------------------------------
   MAIN
   --------------------------------------------------------------------- */
int main(void) {
  
  float v[N]; 
  float r[N]; 
  //float rc[N];
  float m[64];

  float *dv; 
  float *dr; 
  float *dm;
  
  for (int i = 0; i < N; i ++) {
    v[i] = 1.0; 
    r[i] = 7.0;
  }

  cudaMalloc((void**)&dv,N*sizeof(float)); 
  cudaMalloc((void**)&dr,N*sizeof(float));
  cudaMalloc((void**)&dm,64*sizeof(float));
  
  cudaMemcpy(dv,v,N*sizeof(float),cudaMemcpyHostToDevice);
  
  //kernel<<<1,32,32*3*(sizeof(float))>>>(dv,dr,dm);
  //kernel<<<1,16,32*2*(sizeof(float))>>>(dv,dr,dm);
  kernel<<<1,32,64*2*(sizeof(float))>>>(dv,dr,dm);
  

  cudaMemcpy(r,dr,N*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(m,dm,64*sizeof(float),cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i ++) { 
    printf("%f ",r[i]);
  }

  
  printf("\n ------ \n");
  
  for (int i = 0; i < 64; i ++) { 
    printf("%f ",m[i]);
  }
  
  

  //seqScan(v,rc,N);
  //int s = compare(rc,r,0.01,N);
 
 

  //printf ("\n%s\n", s? "same" : "not the same");
  

  return 0;
}


