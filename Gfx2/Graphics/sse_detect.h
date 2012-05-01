// C99

#ifndef __SSE_DETECT_H 
#define __SSE_DETECT_H 

#define cpuid(func,ax,bx,cx,dx)\
        __asm__ __volatile__ ("cpuid":\
        "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));


// on return v and m contains largest supported sse version
void sse_version(int *v, int *m) {
  
  int shifts[5] = {25,26,1,19,20}; 
  
  int a,b,c,d; 
  
  cpuid(1,a,b,c,d); 

  for (int i = 1; i < 6; i ++) {
    if (i < 2) { 
      if (d & (1 << shifts[i-1]))  *v = i; 
    }
    else
      if (c & (1 << shifts[i-1])) {
	*v = i > 4 ? 4 : i; 
	*m = *m + i >= 4 ? 1 : 0;
      }
  }
}



#endif 
