#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <stdint.h> 
#include <stdio.h>


void performExperiment(int size ) { 

  thrust::host_vector<float> values(size);


  //can I fread directly into values ? 
  for (int i = 0; i < size; ++i) {
    values[i] = 1.0;
  }
   
  thrust::device_vector<float> dvalues = values;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  thrust::inclusive_scan(dvalues.begin(),dvalues.end(),dvalues.begin());

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // std::cout << std::endl;

  printf("%f\n", elapsedTime);

  //for (int i = 0; i < size; ++i) {
  values[0] = dvalues[size-1];
  //}
  
  
  //for (int i = 0; i < 512; ++i) {
  printf("%f ",values[0]);
  //}
   


} 


int main(int argc, char **argv){
 
  performExperiment(4096*4096); 
  
  return 0;
}
