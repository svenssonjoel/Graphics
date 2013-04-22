#include <cuda.h>
#include <stdio.h>
#include <stdint.h> 


// For comparisons
//#include "seqScan.c"


/* ------------------------------------------------------------------------
   Unrolled in-place(shared memory) Scan without syncs (16 threads, 32 elts) 
   --------------------------------------------------------------------- */
__device__ int skl_scan(int i, float* input, float *output, uint8_t *sbase, float *maxs) { 
  
  float *s_data = (float*)sbase;
  
  int tid = threadIdx.x; 
  int tids = tid << 1; 

  int eltOffs = blockIdx.x * 32 + tid; 

  // Load data from  global memory into shared memory (in two separate load ops) 
  s_data[tid] = input[eltOffs]; 
  s_data[tid + 16] = input[eltOffs + 16];
    
  // NO SYNC HERE 

  s_data[tids | 1] += s_data[tids]; 
  s_data[(tids | 3) - (tid & 1)] += s_data[tids & 0xFFFFFFFC | 1]; 
  s_data[(tids | 7) - (tid & 3)] += s_data[tids & 0xFFFFFFF8 | 3]; 
  s_data[(tids | 15) - (tid & 7)] += s_data[tids & 0xFFFFFFF0 | 7]; 
  s_data[(tids | 31) - (tid & 15)] += s_data[tids & 0xFFFFFFE0 | 15]; 
 
  // NO Interleaved SYNCS here.

  output[eltOffs] = s_data[tid]; 
  output[eltOffs + 16] = s_data[tid + 16];
  
  if(tid == 0) 
    maxs[i] = s_data[31];

  return 0;
  
}

// Replace with handcoded scan that does not do conditionals.
// This already exists in the scan work I did with Erik. 
/*
__device__ int sklansky(int i, float* input0, float *output0, uint8_t *sbase,float *maxs) {

    uint32_t t2 = ((blockIdx.x*32)+((threadIdx.x&4294967294)|(threadIdx.x&1)));
    uint32_t t9 = ((threadIdx.x&4294967292)|(threadIdx.x&3));
    uint32_t t14 = ((threadIdx.x&4294967288)|(threadIdx.x&7));
    uint32_t t19 = ((threadIdx.x&4294967280)|(threadIdx.x&15));
    ((float*)sbase)[threadIdx.x] = (((threadIdx.x&1)<1) ? input0[t2] : (input0[((blockIdx.x*32)+((threadIdx.x&4294967294)|0))]+input0[t2]));
    //__syncthreads();
    ((float*)(sbase+128))[threadIdx.x] = (((threadIdx.x&3)<2) ? ((float*)sbase)[t9] : (((float*)sbase)[((threadIdx.x&4294967292)|1)]+((float*)sbase)[t9]));
    //__syncthreads();
    ((float*)sbase)[threadIdx.x] = (((threadIdx.x&7)<4) ? ((float*)(sbase+128))[t14] : (((float*)(sbase+128))[((threadIdx.x&4294967288)|3)]+((float*)(sbase+128))[t14]));
    //__syncthreads();
    ((float*)(sbase+128))[threadIdx.x] = (((threadIdx.x&15)<8) ? ((float*)sbase)[t19] : (((float*)sbase)[((threadIdx.x&4294967280)|7)]+((float*)sbase)[t19]));
    //__syncthreads();
    ((float*)sbase)[threadIdx.x] = ((threadIdx.x<16) ? ((float*)(sbase+128))[threadIdx.x] : (((float*)(sbase+128))[15]+((float*)(sbase+128))[threadIdx.x]));
    //__syncthreads();
    
    output0[((blockIdx.x*32)+threadIdx.x)] = ((float*)sbase)[threadIdx.x];
 
    if (threadIdx.x == 0) 
      maxs[i] = ((float*)sbase)[31];
    
    return 0;
} 
*/

/* ------------------------------------------------------------------------
   The Scan kernel (Thousand(s?) of elements! NO SYNCS AT ALL) 
   --------------------------------------------------------------------- */
__global__ void kernel(float* input0,
                       float* output0, 
                       float* maxout){
   
  extern __shared__ __attribute__ ((aligned(16))) uint8_t sbase[];
  
  //  float *maxs = (float*)(sbase+(sizeof(float)*64));
  float *maxs = (float*)(sbase+(sizeof(float)*32));
  
  for (int i = 0; i < 32; i ++) { 
    //sklansky(i,input0+i*32,output0+i*32,sbase,maxs);
    skl_scan(i,input0+i*32,output0+i*32,sbase,maxs);
  }
  
  float v; //  discard this value
  //sklansky(0,maxs,maxs,sbase,&v);
  skl_scan(0,maxs,maxs,sbase,&v);
  

  // distribute (now in two phases) 

  // 15 thread pass
  if (threadIdx.x > 0) {
    for (int j = 0; j < 32; j ++) {
      output0[(blockIdx.x*32)+(threadIdx.x*32+j)] += maxs[threadIdx.x-1];
    }
  }

  // 16 thread pass 
  for (int j = 0; j < 32; j ++) {
    output0[((blockIdx.x+16)*32)+(threadIdx.x*32+j)] += maxs[threadIdx.x+15];
  }
  
  maxout[threadIdx.x] = maxs[threadIdx.x];
  maxout[threadIdx.x+16] = maxs[threadIdx.x+16];

}

#define N 32*32

int main(void) {
  
  float v[N]; 
  float r[N]; 
  //float rc[N];
  float m[32];

  float *dv; 
  float *dr; 
  float *dm;
  
  for (int i = 0; i < N; i ++) {
    v[i] = 1.0; 
    r[i] = 7.0;
  }

  cudaMalloc((void**)&dv,N*sizeof(float)); 
  cudaMalloc((void**)&dr,N*sizeof(float));
  cudaMalloc((void**)&dm,32*sizeof(float));
  
  cudaMemcpy(dv,v,N*sizeof(float),cudaMemcpyHostToDevice);
  
  //kernel<<<1,32,32*3*(sizeof(float))>>>(dv,dr,dm);
  kernel<<<1,16,32*2*(sizeof(float))>>>(dv,dr,dm);


  cudaMemcpy(r,dr,N*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(m,dm,32*sizeof(float),cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i ++) { 
    printf("%f ",r[i]);
  }

  
  printf("\n ------ \n");
  
  for (int i = 0; i < 32; i ++) { 
    printf("%f ",m[i]);
  }
  
  

  //seqScan(v,rc,N);
  //int s = compare(rc,r,0.01,N);
 
 

  //printf ("\n%s\n", s? "same" : "not the same");
  

  return 0;
}



/* 
__global__ void kernel(float* input0,
                       float* output0, 
                       float* maxout){
   
  extern __shared__ __attribute__ ((aligned(16))) uint8_t sbase[];
  
  float *maxs = (float*)(sbase+(sizeof(float)*64));
  
  for (int i = 0; i < 32; i ++) { 

    sklansky(i,input0+i*32,output0+i*32,sbase,maxs);
  }
  
  float v; //  discard this value
  sklansky(0,maxs,maxs,sbase,&v);
  

  // distribute 
  if (threadIdx.x > 0) {
    for (int j = 0; j < 32; j ++) {
      output0[(blockIdx.x*32)+(threadIdx.x*32+j)] += maxs[threadIdx.x-1];
      
    }
  }
  
  maxout[threadIdx.x] = maxs[threadIdx.x];

}
*/
