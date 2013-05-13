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

__global__ void kernel(uint8_t* output0){
  
    float v3;
    float v2;
    uint32_t v1;
    v3 = 0.0f;
    v2 = 0.0f;
    v1 = 1;
    while (((((v3*v3)+(v2*v2))<4.0f)&&(v1<512))){
      
        float t6;
        float t5;
        uint32_t t4;
        t6 = v3;
        t5 = v2;
        t4 = v1;
        v3 = (((t6*t6)-(t5*t5))+(-2.0f+(((float)threadIdx.x)*6.25e-3f)));
        v2 = (((2.0f*t6)*t5)+(1.2f-(((float)blockIdx.x)*4.6875e-3f)));
        v1 = (t4+1);
        
    }
    output0[((blockIdx.x*512)+threadIdx.x)] = ((((uint8_t)v1)%16)*16);
    
  
}

__global__ void plate1(uint8_t* output0){
  
    float v3;
    float v2;
    uint32_t v1;
    v3 = 0.0f;
    v2 = 0.0f;
    v1 = 1;
    while (((((v3*v3)+(v2*v2))<4.0f)&&(v1<512))){
      
        float t6;
        float t5;
        uint32_t t4;
        t6 = v3;
        t5 = v2;
        t4 = v1;
        v3 = (((t6*t6)-(t5*t5))+(-0.69106f+(((float)threadIdx.x)*3.008172e-7f)));
        v2 = (((2.0f*t6)*t5)+(0.387228f-(((float)blockIdx.x)*2.4418114e-7f)));
        v1 = (t4+1);
        
    }
    output0[((blockIdx.x*512)+threadIdx.x)] = ((((uint8_t)v1)%16)*16);
    
  
}

__global__ void plate2(uint8_t* output0){
  
    float v3;
    float v2;
    uint32_t v1;
    v3 = 0.0f;
    v2 = 0.0f;
    v1 = 1;
    while (((((v3*v3)+(v2*v2))<4.0f)&&(v1<512))){
      
        float t6;
        float t5;
        uint32_t t4;
        t6 = v3;
        t5 = v2;
        t4 = v1;
        v3 = (((t6*t6)-(t5*t5))+(-0.793114f+(((float)threadIdx.x)*1.3693166e-4f)));
        v2 = (((2.0f*t6)*t5)+(0.140974f-(((float)blockIdx.x)*2.0146875e-4f)));
        v1 = (t4+1);
        
    }
    output0[((blockIdx.x*512)+threadIdx.x)] = ((((uint8_t)v1)%16)*16);
    
  
}
__global__ void plate3(uint8_t* output0){
  
    float v3;
    float v2;
    uint32_t v1;
    v3 = 0.0f;
    v2 = 0.0f;
    v1 = 1;
    while (((((v3*v3)+(v2*v2))<4.0f)&&(v1<512))){
      
        float t6;
        float t5;
        uint32_t t4;
        t6 = v3;
        t5 = v2;
        t4 = v1;
        v3 = (((t6*t6)-(t5*t5))+(-0.745464f+(((float)threadIdx.x)*1.4854595e-7f)));
        v2 = (((2.0f*t6)*t5)+(0.11303f-(((float)blockIdx.x)*1.23051e-7f)));
        v1 = (t4+1);
        
    }
    output0[((blockIdx.x*512)+threadIdx.x)] = ((((uint8_t)v1)%16)*16);
    
  
}


__global__ void mandel(uint8_t *out) { 
  
  int bid = blockIdx.x; 
  int tid = threadIdx.x; 

  float x = 0.0, y = 0.0, xsq = 0.0, ysq = 0.0;
  int color = 1; 

  while (color < ITERS && (xsq + ysq) < max_size) {

    xsq = x*x;
    ysq = y*y;
    y = 2*x*y+(ymax - blockIdx.x*deltaQ);
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

  //mandel<<<HEIGHT,WIDTH,0>>>(dr);
  //kernel<<<HEIGHT,WIDTH,0>>>(dr);	
  //plate1<<<HEIGHT,WIDTH,0>>>(dr);	
  //plate2<<<HEIGHT,WIDTH,0>>>(dr);	  		
  plate3<<<HEIGHT,WIDTH,0>>>(dr);	  		

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


