#include <cuda.h>
#include <stdio.h>



__global__ void scan_local(float *in, float *out) {
  out[0] = in[0];
  for (int i = 0; i < 32; i ++) {
    
    out[i] = out[i-1] + in[i]; 
  }

    
} 




int main(void) {
  
  float v[32]; 
  float r[32]; 

  float *dv; 
  float *dr; 

  for (int i = 0; i < 32; i ++) {
    v[i] = 1.0; 
    r[i] = 7.0;
  }

  cudaMalloc((void**)&dv,32*sizeof(float)); 
  cudaMalloc((void**)&dr,32*sizeof(float));
  
  cudaMemcpy(dv,v,32*sizeof(float),cudaMemcpyHostToDevice);
  
  scan_local<<<1,1,0>>>(dv,dr);

  cudaMemcpy(r,dr,32*sizeof(float),cudaMemcpyDeviceToHost);

  for (int i = 0; i < 32; i ++) { 
    printf("%f ",r[i]);
  }

  return 0;
}
