
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

int main(void) {

  Mat4x4 test;
  Mat4x4 a;
  Mat4x4 b; 
  
  loadIdentity4x4(a);
  loadIdentity4x4(b);
  zero4x4(test);

  printf("test\n");
  for (int i = 0; i < 16; i ++) {
    printf("%s",(i % 4 == 0 && i ? "\n" : ""));
    printf("%f ",test[i]);
  }

  printf("\na\n");
  for (int i = 0; i < 16; i ++) {
    printf("%s",(i % 4 == 0 && i ? "\n" : ""));
    printf("%f ",a[i]);
  }

  printf("\nb\n");
  for (int i = 0; i < 16; i ++) {
    printf("%s",(i % 4 == 0 && i ? "\n" : ""));
    printf("%f ",b[i]);
  }
  
  mul4x4(test,a,b);
  
  printf("\ntest = a * b\n");
  for (int i = 0; i < 16; i ++) {
    printf("%s",(i % 4 == 0 && i ? "\n" : ""));
    printf("%f ",b[i]);
  }

  translate4x4(test,3,2,1);
  
  printf("\ntranslate(test,3,2,1)\n");
  for (int i = 0; i < 16; i ++) {
    printf("%s",(i % 4 == 0 && i ? "\n" : ""));
    printf("%f ",test[i]);
  }

  scale4x4(test,2,2,2);
  
  printf("\nscale(test,2,2,2)\n");
  for (int i = 0; i < 16; i ++) {
    printf("%s",(i % 4 == 0 && i ? "\n" : ""));
    printf("%f ",test[i]);
  }

  loadIdentity4x4(test);
  rotate4x4(test,3.14,0,0,1);
  
  printf("\nrotate(test,rad,1,0,0)\n");
  for (int i = 0; i < 16; i ++) {
    printf("%s",(i % 4 == 0 && i ? "\n" : ""));
    printf("%f ",test[i]);
  }

}
