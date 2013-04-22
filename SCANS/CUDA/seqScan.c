
int seqScan(float *in, float *out, int n) { 

  out[0] = in[0]; 

  for (int i = 1; i < n; i ++) {
    out[i] = out[i-1] +  in[i]; 
    
  }
  return 0;
} 



int compare(float *a, float *b, float y, int n) {
  
  int same = 1; 

  for (int i = 0; i < n; i ++) 
    if (fabs(a[i] - b[i]) > y) same = 0;  
  

  return same; 

} 
