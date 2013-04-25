#include <cuda.h>
#include <stdio.h>
#include <stdint.h> 



// For comparisons
//#include "seqScan.c"


#define CLONES 8
#define REPS   8

// block size in ELEMENTS!
#define BLOCK_SIZE (CLONES*REPS*64) 

//#define N 4096*CLONES*REPS*64

#define N (BS*BOS*SEQUENTIAL_SIZE)
#define BS 128
#define SEQUENTIAL_SIZE 512
#define BLOCK_DATA_SIZE (BS * SEQUENTIAL_SIZE)
#define BOS 100
#define WARPSIZE 32

/* ------------------------------------------------------------------------
   Reduction kernel (reduce 64 element sub blocks of global array)  
   --------------------------------------------------------------------- */
  
__global__ void reduce(float *input, float* output) { 
    
  extern __shared__ float s_data[];
  
  int tid = threadIdx.x; 
  
  int warp = tid / 32;
  int wid  = tid % 32; 
  
  float sum = 0.0; 
  
  /* Eight element sequential reduction */
  for (int i = 0; i < SEQUENTIAL_SIZE; i ++) { 
    sum += input[blockIdx.x*BLOCK_DATA_SIZE+(tid*SEQUENTIAL_SIZE) + i]; 
  } 
  
  s_data[warp*32+wid] = sum;
  
  /* Perform (many) 8 element parallel reduction */ 
  
  // BS threads 
  // BS/32 warps (each doing 4 reductions (8 elements to 1) !)
  if (wid < 16) 
    s_data[warp*32+(wid<<1)] += s_data[warp*32+(wid<<1)+1];

  if (wid < 8) 
    s_data[warp*32+(wid<<2)] += s_data[warp*32+(wid<<2)+2];
  
  if (wid < 4) 
    s_data[warp*32+(wid<<3)] += s_data[warp*32+(wid<<3)+4];
    
  // at this point we have (BS/32)*4  element reductions

  // chunking out the 4 reductions per warp
  if (wid < 4)
    output[blockIdx.x*(4*BS/WARPSIZE)+warp*4+wid] = s_data[warp*32+(wid<<3)];

} 


/* ------------------------------------------------------------------------
   Unrolled and in-place sklansky kernel  
   --------------------------------------------------------------------- */
__device__ void skl_scan(int i, 
			 float* input, 
			 float *output, 
			 float *s_data, // The shared memory
			 float *maxs) { 
    
  int tid = threadIdx.x; 
  int tids = tid << 1; 

  // Load data from global memory into shared memory (in two separate load ops)
  s_data[tids] = input[tids]; 
  s_data[tids+1] = input[tids+1]; 
  //  __syncthreads();


  s_data[tids | 1] += s_data[tids]; 
  s_data[(tids | 3) - (tid & 1)] += s_data[tids & 0xFFFFFFFC | 1]; 
  s_data[(tids | 7) - (tid & 3)] += s_data[tids & 0xFFFFFFF8 | 3]; 
  s_data[(tids | 15) - (tid & 7)] += s_data[tids & 0xFFFFFFF0 | 7]; 
  s_data[(tids | 31) - (tid & 15)] += s_data[tids & 0xFFFFFFE0 | 15]; 
  s_data[(tids | 63) - (tid & 31)] += s_data[tids & 0xFFFFFFC0 | 31];
  // NO Interleaved SYNCS here.

  //__syncthreads();
  output[tids] = s_data[tids]; 
  output[tids+1] = s_data[tids+1];
  
  //__syncthreads();
  if(tid % 32 == 0) 
    maxs[(i<<3)+(tid>>5)] = s_data[(tid << 1) | 0x3F];
  //maxs[i*CLONES+(tid / 32)] = s_data[(tid / 32)*64 + 63];
  // (i<<3)+(tid>>5)                    ((tid>>5)<<6) + 63 
  //                                    (tid << 1) | 0x3F)   
  
}


/* ------------------------------------------------------------------------
   The Scan kernel (Thousand(s) of elements) 
   --------------------------------------------------------------------- */
__global__ void kernel(float* input0,
                       float* output0, 
                       float* maxout){
   
  // shared data. (two different kinds. warp local and across warps.) 
  extern __shared__ float s_data[]; 
  float *maxs = &s_data[512]; 
  
  // Sequentially execute 64 scans
  for (int i = 0; i < REPS; i ++) {  
    skl_scan(i,
	     input0+(blockIdx.x*BLOCK_SIZE)+(i*512),
	     output0+(blockIdx.x*BLOCK_SIZE)+(i*512),
	     s_data,maxs);
  }

  // Now needs one __syncthreads() here! 
  __syncthreads();

  // in parallel scan the maximum array 
  float v; //discard this value.
  if (threadIdx.x < 32) 
    skl_scan(0,maxs,maxs,(float *)s_data,&v);
  
  __syncthreads();

  
  // really messy code  
  for (int j = 0; j < REPS; j ++) {
    if (j != 0 || threadIdx.x >=  64) 
      output0[(blockIdx.x*BLOCK_SIZE)+(j*256)+threadIdx.x] += maxs[(((j*256)+threadIdx.x) / 64)-1];
    output0[(blockIdx.x*BLOCK_SIZE)+(j*256)+threadIdx.x+2048] += maxs[(((j*256)+threadIdx.x+2048) /64)-1];
  }
 
  // This is a debug step. 
  if (threadIdx.x < 32) {
    maxout[threadIdx.x] = maxs[threadIdx.x];
    maxout[threadIdx.x+32] = maxs[threadIdx.x+32];
  }
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
  
  //for (int i = 0; i < N; i ++) {
  //  v[i] = 1.0; 
  //  r[i] = 7.0;
  //}
  for (int i = 0; i < N/8; i ++) {
    for (int j = 0; j < 8; j ++) { 
      v[i*8+j] = 1; 
      r[i*8+j] = 7.0;
    }
  }
  

  cudaMalloc((void**)&dv,N*sizeof(float)); 
  cudaMalloc((void**)&dr,N*sizeof(float));
  cudaMalloc((void**)&dm,64*sizeof(float));
  
  cudaMemcpy(dv,v,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemset(dr,0,N*sizeof(float));

  //kernel<<<1,32,32*3*(sizeof(float))>>>(dv,dr,dm);
  //kernel<<<1,16,32*2*(sizeof(float))>>>(dv,dr,dm);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //kernel<<<4096,256,(512+64)*(sizeof(float))>>>(dv,dr,dm);

  reduce<<<BOS,BS,BS*sizeof(float)>>>(dv,dr);
  
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // std::cout << std::endl;

  cudaMemcpy(r,dr,N*sizeof(float),cudaMemcpyDeviceToHost);
  //cudaMemcpy(m,dm,64*sizeof(float),cudaMemcpyDeviceToHost);


  for (int i = 0; i < 129 /*N*/; i ++) { 
    printf("%f ",r[i]);
  }

  int count = 0; 
  for (int i = 0; i < N; i ++) { 
    if (!(r[i] > 0.0)) break;
    count ++;
  }

  printf("number of results %d\n", count);
  
  
  //printf("\n ------ \n");
  
  //for (int i = 0; i < 64; i ++) { 
  //  printf("%f ",m[i]);
  //}
  
  

  printf("Elapsed time: %f\n", elapsedTime);


  //seqScan(v,rc,N);
  //int s = compare(rc,r,0.01,N);
 
 

  //printf ("\n%s\n", s? "same" : "not the same");
  

  return 0;
}


