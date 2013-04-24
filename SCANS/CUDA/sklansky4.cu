#include <cuda.h>
#include <stdio.h>
#include <stdint.h> 


// For comparisons
//#include "seqScan.c"

#define ELTS 64
#define BS 1024
#define N 16384*BS


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

  // Load data from global memory into shared memory (in two separate load ops)
  s_data[tid] = input[tid]; 
  s_data[tid + 32] = input[tid + 32]; 
  // NO SYNC HERE 

  s_data[tids | 1] += s_data[tids]; 
  s_data[(tids | 3) - (tid & 1)] += s_data[tids & 0xFFFFFFFC | 1]; 
  s_data[(tids | 7) - (tid & 3)] += s_data[tids & 0xFFFFFFF8 | 3]; 
  s_data[(tids | 15) - (tid & 7)] += s_data[tids & 0xFFFFFFF0 | 7]; 
  s_data[(tids | 31) - (tid & 15)] += s_data[tids & 0xFFFFFFE0 | 15]; 
  s_data[(tids | 63) - (tid & 31)] += s_data[tids & 0xFFFFFFC0 | 31];
  // NO Interleaved SYNCS here.

  output[tid] = s_data[tid]; 
  output[tid + 32] = s_data[tid + 32];
  
  if(tid == 0) 
    maxs[i] = s_data[63];
  
}

__device__ void skl_scan16(int i, 
			   float* input, 
			   float *output, 
			   float *s_data) { 
    
  int tid = threadIdx.x; 
  int tids = tid << 1; 

  // Load data from global memory into shared memory (in two separate load ops)
  s_data[tid] = input[tid]; 
  s_data[tid + 8] = input[tid + 8]; 
  // NO SYNC HERE 

  s_data[tids | 1] += s_data[tids]; 
  s_data[(tids | 3) - (tid & 1)] += s_data[tids & 0xFFFFFFFC | 1]; 
  s_data[(tids | 7) - (tid & 3)] += s_data[tids & 0xFFFFFFF8 | 3]; 
  s_data[(tids | 15) - (tid & 7)] += s_data[tids & 0xFFFFFFF0 | 7]; 
  //s_data[(tids | 31) - (tid & 15)] += s_data[tids & 0xFFFFFFE0 | 15]; 
  //s_data[(tids | 63) - (tid & 31)] += s_data[tids & 0xFFFFFFC0 | 31];
  // NO Interleaved SYNCS here.

  output[tid] = s_data[tid]; 
  output[tid + 8] = s_data[tid + 8];
    
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
  for (int i = 0; i < 16; i ++) { 
    skl_scan(i,
	     (input0+blockIdx.x*BS)+i*64,
	     (output0+blockIdx.x*BS)+i*64,
	     s_data,maxs);
  }
  
  // in parallel scan the maximum array 
  //float v; //discard this value.

  if (threadIdx.x < 16)
    skl_scan16(0,maxs,maxs,(float *)s_data);
  

  // distribute

  // 15 thread pass. 
  if (threadIdx.x > 0 && threadIdx.x < 16) {
    for (int j = 0; j < 64; j ++) {
      output0[(blockIdx.x*BS)+(threadIdx.x*64)+j] += maxs[threadIdx.x-1];
    }
  }

  // 32 thread pass. 
  //for (int j = 0; j < 64; j ++) {
  //  output0[(blockIdx.x*BS)+((threadIdx.x+32)*64)+j] += maxs[threadIdx.x+31];
  //}

  // This is a debug step. 
  //maxout[threadIdx.x] = maxs[threadIdx.x];
  //maxout[threadIdx.x+32] = maxs[threadIdx.x+32];

}

/* ------------------------------------------------------------------------
   MAIN
   --------------------------------------------------------------------- */
int main(void) {
  
  float *v; 
  float *r; 
  //float rc[N];
  float m[64];

  float *dv; 
  float *dr; 
  float *dm;
  
  v = (float*)malloc(sizeof(float) * N);
  r = (float*)malloc(sizeof(float) * N);
  memset(m,0,64*sizeof(float));
  
  for (int i = 0; i < N; i ++) {
    v[i] = 1.0; 
    r[i] = 7.0;
  }

  cudaMalloc((void**)&dv,N*sizeof(float)); 
  cudaMalloc((void**)&dr,N*sizeof(float));
  cudaMalloc((void**)&dm,4096*64*sizeof(float));
  
  cudaMemcpy(dv,v,N*sizeof(float),cudaMemcpyHostToDevice);
  
  //kernel<<<1,32,32*3*(sizeof(float))>>>(dv,dr,dm);
  //kernel<<<1,16,32*2*(sizeof(float))>>>(dv,dr,dm);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  kernel<<<16384,32,64*2*(sizeof(float))>>>(dv,dr,dm);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // std::cout << std::endl;

  cudaMemcpy(r,dr,N*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(m,dm,64*sizeof(float),cudaMemcpyDeviceToHost);


  for (int i = 0; i < 256; i ++) { 
    printf("%f ",r[i]);
  }

  
  printf("\n ------ \n");
  
  for (int i = 0; i < 64; i ++) { 
    printf("%f ",m[i]);
  }
  
  

  printf("Elapsed time: %f\n", elapsedTime);


  //seqScan(v,rc,N);
  //int s = compare(rc,r,0.01,N);
 
 

  //printf ("\n%s\n", s? "same" : "not the same");
  

  return 0;
}


