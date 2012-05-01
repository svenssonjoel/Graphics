/* matrix.c */

/*
  TODO: 
    frustum(Mat4x4 r, 
            float left, 
	    float right, 
	    float bottom, 
	    float top, 
	    float near, 
	    float far)
*/
#include "matrix.h"

#include <stdlib.h> 
#include <stdio.h>
#include <memory.h>
#include <math.h>


/* --------------------------------------------------------------------------
   openGL matrices
   -------------------------------------------------------------------------- */ 
/*
  [ 0][ 4][ 8][12] 
  [ 1][ 5][ 9][13] gl matrix is a flat array indexed 
  [ 2][ 6][10][14] like this. 
  [ 3][ 7][11][15]

  
Translation: 
  [ 0][ 4][ 8][ X]
  [ 1][ 5][ 9][ Y]
  [ 2][ 6][10][ Z]
  [ 3][ 7][11][15]

Scale: 
  [ X][ 4][ 8][12]
  [ 1][ Y][ 9][13]
  [ 2][ 6][ Z][14]
  [ 3][ 7][11][15]
  
*/



/* --------------------------------------------------------------------------
   zero
   -------------------------------------------------------------------------- */ 

void zero4x4(Mat4x4 input) {
  memset(input,0,sizeof(Mat4x4));
}

/* --------------------------------------------------------------------------
   identity
   -------------------------------------------------------------------------- */ 

void loadIdentity4x4(Mat4x4 input) {
  memset(input,0,sizeof(Mat4x4));
  input[0] = input[5] = input[10] = input[15] = 1.0; 
}


/* --------------------------------------------------------------------------
   mul4x4
   -------------------------------------------------------------------------- */ 

/* Optimize this! */
void mul4x4(Mat4x4 r, const  Mat4x4 a, const Mat4x4 b)
{
  int i;

  Mat4x4 tmp;

  for (i = 0; i < 16; i++) {
      tmp[i] = 0.0f;
		
      for(int k = 0; k < 4; k++) {
	tmp[i] += a[(i%4)+k*4] * b[k+(i/4)*4];
      }
    }
		
  for (i = 0; i < 16; i++) {
      r[i] = tmp[i];
    }
}

/* --------------------------------------------------------------------------
   translate4x4
   -------------------------------------------------------------------------- */ 
void translate4x4(Mat4x4 r, float x, float y, float z) {
  
  Mat4x4 tmp; 

  loadIdentity4x4(tmp);
  
  tmp[12] = x;
  tmp[13] = y; 
  tmp[14] = z; 
  
  mul4x4(r,r,tmp);
}

/* --------------------------------------------------------------------------
   scale4x4
   -------------------------------------------------------------------------- */ 
void scale4x4(Mat4x4 r, float x, float y, float z) {
  
  Mat4x4 tmp; 

  loadIdentity4x4(tmp);
  
  tmp[0] = x;
  tmp[5] = y; 
  tmp[10] = z; 
  
  mul4x4(r,r,tmp);
}

/* --------------------------------------------------------------------------
   rotate4x4
   -------------------------------------------------------------------------- */ 
/* Optimize this! */
void rotate4x4(Mat4x4 r,float rad, float x,float y, float z) {
  Mat4x4 mx; 
  Mat4x4 my;
  Mat4x4 mz;

  loadIdentity4x4(mx);
  loadIdentity4x4(my);
  loadIdentity4x4(mz);

  if (x){
    mx[5] = mx[10] = x * cosf(rad);
    mx[6] = x * (-sinf(rad));
    mx[9] = x * sinf(rad);
  }
  if (y) {
    my[0] = my[10] = y * cosf(rad);
    my[2] = y * sinf(rad);
    my[8] = y * (-sinf(rad));
  }
  if (z) {
    mz[0] = mz[5] = z * cosf(rad);
    mz[1] = z * (-sinf(rad));
    mz[4] = z * sinf(rad);
  }
  
  mul4x4(r,r,mx);
  mul4x4(r,r,my);
  mul4x4(r,r,mz);
}



/* --------------------------------------------------------------------------
   Projection matrices 
   -------------------------------------------------------------------------- */ 


/* --------------------------------------------------------------------------
   Orthographic projection 
   -------------------------------------------------------------------------- */ 
void setOrtho(Mat4x4 m, float l, float r, float b, float t, float n, float f) {

  zero4x4(m); // start with empty

  m[ 0] = 2.0f / (r - l);
  m[ 5] = 2.0f / (t - b); 
  m[10] = -2.0f / (f - n); 
  m[12] = -(r + l) / (r - l);
  m[13] = -(t + b) / (t - b); 
  m[14] = -(f + n) / (f - n); 
  m[15] = 1.0f;
}

/*
  [ 0][ 4][ 8][12] 
  [ 1][ 5][ 9][13] gl matrix is a flat array indexed 
  [ 2][ 6][10][14] like this. 
  [ 3][ 7][11][15]
*/



void setOrtho2D(Mat4x4 m, float l, float r, float b, float t) {
  setOrtho(m,l,r,b,t,-1.0f,1.0f);
}

  
