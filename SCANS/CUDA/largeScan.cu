#include <cuda.h>
#include <stdio.h>
#include <stdint.h> 

#include <sm_30_intrinsics.h>

#define CLONES 8
#define REPS   8

// block size in ELEMENTS!
#define BLOCK_SIZE (CLONES*REPS*64) 

//#define N 4096*CLONES*REPS*64

#define N 4096*4096 // (BS*BOS*SEQUENTIAL_SIZE)
#define BS 512
#define SEQUENTIAL_SIZE 32
#define BLOCK_DATA_SIZE (BS * SEQUENTIAL_SIZE)
#define BOS 100
#define WARPSIZE 32


/* ------------------------------------------------------------------------

   --------------------------------------------------------------------- */
__global__ void reduce(float *input, float* output) { 
  extern __shared__ float s_data[];
  
  int tid = threadIdx.x; 
  int laneId = tid & 0x1f;
  int warp   = tid / 32;

  float sum = 0.0;
  for (int i = 0; i < 16; i ++) { 
    //im not not 100% sure here
    sum += input[blockIdx.x*4096+tid+i*16];
  }
 
  float value; 

  // each warp performs a 32 element reductions using shfls
  for (int i = 16; i > 0; i = i / 2) { 
    value = __shfl_down(sum,i);
    sum += value; 
  }

  // shared memory to communicate between warps
  if (laneId == 0) 
    s_data[warp] = sum; 
  
  // the only sync 
  __syncthreads();    
  
  // Now 8 elements left
  // Process the last 8 elements in one warp
  if (warp == 0 && laneId < 8) { 
    
    sum = s_data[laneId];
    

    // 3 step 8 to 1 value reduction
    value = __shfl_down(sum,4);
    sum += value; 
      
    value = __shfl_down(sum,2);
    sum += value; 

    value = __shfl_down(sum,1);
    sum += value; 
  
    
    if(laneId == 0) 
      output[blockIdx.x] = sum;
    
  } 

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
                       float* maxins){
   
  // shared data. (two different kinds. warp local and across warps.) 
  extern __shared__ float s_data[]; 
  float *maxs = &s_data[512]; 
  
  // Each block add the reduction result to first element of block 
  if (blockIdx.x > 0 && threadIdx.x == 0) {
    input0[blockIdx.x*4096] += maxins[blockIdx.x];
  }
  __syncthreads(); 
  
  // Sequentially execute REPS number of  scans
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
      output0[(blockIdx.x*BLOCK_SIZE)+(j*256)+threadIdx.x] += 
        maxs[(((j*256)+threadIdx.x) / 64)-1];
      output0[(blockIdx.x*BLOCK_SIZE)+(j*256)+threadIdx.x+2048] += 
        maxs[(((j*256)+threadIdx.x+2048) /64)-1];
  }
 
  // This is a debug step. 
  //if (threadIdx.x < 32) {
  //  maxout[threadIdx.x] = maxs[threadIdx.x];
  //  maxout[threadIdx.x+32] = maxs[threadIdx.x+32];
  //}
}


/* ------------------------------------------------------------------------
   MAIN
   --------------------------------------------------------------------- */
int main(void) {
  
  float *v; 
  float *r; 
  //float rc[N];
  float m[64];

  float *dinput; 
  float *dreduced;
  float *dimscan; 
  float *dresult;
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
  

  cudaMalloc((void**)&dinput,N*sizeof(float)); 
  cudaMalloc((void**)&dreduced,4096*sizeof(float)); 
  cudaMalloc((void**)&dimscan,4096*sizeof(float)); 

  cudaMalloc((void**)&dresult,N*sizeof(float));
  cudaMalloc((void**)&dm,64*sizeof(float));
  
  cudaMemcpy(dinput,v,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemset(dresult,0,N*sizeof(float));

  //kernel<<<1,32,32*3*(sizeof(float))>>>(dv,dr,dm);
  //kernel<<<1,16,32*2*(sizeof(float))>>>(dv,dr,dm);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // Perform scan! 
  reduce<<<4096,256,32*sizeof(float)>>>(dinput,dreduced);
  
  kernel<<<1,256,(512+64)*sizeof(float)>>>(dreduced, dimscan, dm);
  
  kernel<<<4096,256,(512+64)*sizeof(float)>>>(dinput, dresult, dimscan);
  

  
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // std::cout << std::endl;

  cudaMemcpy(r,dresult,N*sizeof(float),cudaMemcpyDeviceToHost);
  //cudaMemcpy(m,dm,64*sizeof(float),cudaMemcpyDeviceToHost);


  for (int i = 0; i < 10 /*N*/; i ++) { 
    printf("%f ",r[i]);
  }
  printf("\n");
  for (int i = 0; i < 10 /*N*/; i ++) { 
    printf("%f ",r[4090+i]);
  }
  printf("\n");
  for (int i = 0; i < 10 /*N*/; i ++) { 
    printf("%f ",r[8186+i]);
  }
  printf("\n");
  for (int i = 0; i < 10 /*N*/; i ++) { 
    printf("%f ",r[12282+i]);
  }
  printf("\n");
  for (int i = 0; i < 10 /*N*/; i ++) { 
    printf("%f ",r[4096*4096-10+i]);
  }


  int count = 0; 
  for (int i = 0; i < 4096; i ++) { 
    if (!(r[i] > 0.0)) break;
    count ++;
  }
  printf("\n");
  
  printf("number of results %d\n", count);
  
  printf("Elapsed time: %f\n", elapsedTime);


  return 0;
}
