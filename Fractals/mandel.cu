#include <cuda.h>
#include <stdio.h>
#include <stdint.h> 


#define WIDTH 512
#define HEIGHT 512
#define ITERS 512

#define N (WIDTH*HEIGHT)

#define max_size   4
#define max_colors 16

#define xmax  1.2f
#define xmin -2.0f
#define ymax  1.2f
#define ymin -1.2f

#define deltaP ((xmax - xmin)/512)
#define deltaQ ((ymax - ymin)/512) 

__global__ void mandel(uint8_t *out) { 
  
  int bid = blockIdx.x; 
  int tid = threadIdx.x; 

  float x = 0.0, y = 0.0, xsq = 0.0, ysq = 0.0;
  int color = 1; 

  while (color < ITERS && (xsq + ysq) < max_size) {

    xsq = x*x;
    ysq = y*y;
    y *= x;
    y += y + (ymax - blockIdx.x*deltaQ);
    x = xsq - ysq + (xmin + threadIdx.x * deltaP);
    color ++;
  }
  
  
  out[bid* 512 + tid] = (color % 8) * 32; // % max_colors;

}

/* ------------------------------------------------------------------------
   MAIN
   --------------------------------------------------------------------- */
int main(void) {
  
   
  uint8_t *r; 

  uint8_t *dr; 
  
  r = (uint8_t*)malloc(sizeof(uint8_t) * N);
  
  cudaMalloc((void**)&dr,N*sizeof(uint8_t));

  cudaMemset(dr,0,N*sizeof(uint8_t));
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  mandel<<<HEIGHT,WIDTH,0>>>(dr);
  
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // std::cout << std::endl;

  cudaMemcpy(r,dr,N*sizeof(uint8_t),cudaMemcpyDeviceToHost);
  //cudaMemcpy(m,dm,64*sizeof(float),cudaMemcpyDeviceToHost);


  for (int i = 0; i < N; i ++) { 
    printf("%d ",r[i]);
  }

    
  printf("Elapsed time: %f\n", elapsedTime);


  FILE *file; 
  file = fopen("image.out","w");
  fwrite(r,sizeof(uint8_t),N,file);
  fclose(file);
  

  return 0;
}


