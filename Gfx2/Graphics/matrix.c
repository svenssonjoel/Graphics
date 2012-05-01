/* matrix.c */


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

void matrix_zero4x4f(Mat4x4f input) {
  memset(input,0,sizeof(Mat4x4f));
}

/* --------------------------------------------------------------------------
   identity
   -------------------------------------------------------------------------- */ 

void matrix_identity4x4f(Mat4x4f input) {
  memset(input,0,sizeof(Mat4x4f));
  input[0] = input[5] = input[10] = input[15] = 1.0f; 
}


/* --------------------------------------------------------------------------
   mul4x4
   -------------------------------------------------------------------------- */ 

/* Optimize this! */
void matrix_mul4x4f(Mat4x4f r, const  Mat4x4f a, const Mat4x4f b) {
  int i;

  Mat4x4f tmp;

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

void matrix_transform4x4f(Vector4f *r, Mat4x4f m, Vector4f *v) {
  Vector4f tmp; 
  
  tmp.x = v->x * m[0] + v->y * m[4] + v->z * m[8] + v->w * m[12];
  
  tmp.y = v->x * m[1] + v->y * m[5] + v->z * m[9] + v->w * m[13];
  
  tmp.z = v->x * m[2] + v->y * m[6] + v->z * m[10] + v->w * m[14];
  
  tmp.w = v->x * m[3] + v->y * m[7] + v->z * m[11] + v->w * m[15];
  
  r->x = tmp.x;
  r->y = tmp.y; 
  r->z = tmp.z;
  r->w = tmp.w;
  /* [0 ] [4 ] [8 ] [12 ] 
     [1 ] [5 ] [9 ] [13 ]
     [2 ] [6 ] [10] [14 ]
     [3 ] [7 ] [11] [15 ]  */
}


  

/* --------------------------------------------------------------------------
   translate4x4
   -------------------------------------------------------------------------- */ 
void matrix_translate4x4f(Mat4x4f r, float x, float y, float z) {
  
  Mat4x4f tmp; 

  matrix_identity4x4f(tmp);
  
  tmp[12] = x;
  tmp[13] = y; 
  tmp[14] = z; 
  
  matrix_mul4x4f(r,r,tmp);
}

/* --------------------------------------------------------------------------
   scale4x4
   -------------------------------------------------------------------------- */ 
void matrix_scale4x4f(Mat4x4f r, float x, float y, float z) {
  
  Mat4x4f tmp; 

  matrix_identity4x4f(tmp);
  
  tmp[0] = x;
  tmp[5] = y; 
  tmp[10] = z; 
  
  matrix_mul4x4f(r,r,tmp);
}

/* --------------------------------------------------------------------------
   rotate4x4
   -------------------------------------------------------------------------- */ 
/* Optimize this! */
void matrix_rotate4x4f(Mat4x4f r,float rad, float x,float y, float z) {
  Mat4x4f mx; 
  Mat4x4f my;
  Mat4x4f mz;

  matrix_identity4x4f(mx);
  matrix_identity4x4f(my);
  matrix_identity4x4f(mz);

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
  
  matrix_mul4x4f(r,r,mx);
  matrix_mul4x4f(r,r,my);
  matrix_mul4x4f(r,r,mz);
}



/* --------------------------------------------------------------------------
   Projection matrices 
   -------------------------------------------------------------------------- */ 


/* --------------------------------------------------------------------------
   Orthographic projection 
   -------------------------------------------------------------------------- */ 
void matrix_orthof(Mat4x4f m, float l, float r, float b, float t, float n, float f) {

  matrix_zero4x4f(m); // start with empty

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
void matrix_ortho2Df(Mat4x4f m, float l, float r, float b, float t) {
  matrix_orthof(m,l,r,b,t,1.0f,-1.0f);
}


/* --------------------------------------------------------------------------
   Perspective (frustum)
   -------------------------------------------------------------------------- */ 
  
void matrix_frustumf(Mat4x4f m, float l, float r, float b, float t, float n, float f) {

  matrix_zero4x4f(m); // start with empty

  m[ 0] = (2.0f * n) / (r - l);
  m[ 5] = (2.0f * n) / (t - b); 
  m[ 8] = (r + l) / (r - l);
  m[ 9] = (t + b) / (t - b);
  m[10] = -(f + n) / (f - n);
  m[11] = -1;
  m[14] = -(2.0f * f * n) / (f - n);

}

float matrix_det4x4f(Mat4x4f m) {
  
  float a0 = m[ 0]*m[ 5] - m[ 1]*m[ 4];
  float a1 = m[ 0]*m[ 6] - m[ 2]*m[ 4];
  float a2 = m[ 0]*m[ 7] - m[ 3]*m[ 4];
  float a3 = m[ 1]*m[ 6] - m[ 2]*m[ 5];
  float a4 = m[ 1]*m[ 7] - m[ 3]*m[ 5];
  float a5 = m[ 2]*m[ 7] - m[ 3]*m[ 6];
  float b0 = m[ 8]*m[13] - m[ 9]*m[12];
  float b1 = m[ 8]*m[14] - m[10]*m[12];
  float b2 = m[ 8]*m[15] - m[11]*m[12];
  float b3 = m[ 9]*m[14] - m[10]*m[13];
  float b4 = m[ 9]*m[15] - m[11]*m[13];
  float b5 = m[10]*m[15] - m[11]*m[14];

  float det = a0*b5 - a1*b4 + a2*b3 + a3*b2 - a4*b1 + a5*b0;

  return det;

  /*
  float value =
    m[12] * m[9] * m[6] * m[3]-m[8] * m[13] * m[6] * m[3]-m[12] * m[1] * m[10] * m[3]+m[4] * m[9] * m[10] * m[3]+
    m[8] * m[1] * m[14] * m[3]-m[4] * m[5] * m[14] * m[3]-m[12] * m[5] * m[2] * m[7]+m[8] * m[9] * m[2] * m[7]+
    m[12] * m[1] * m[10] * m[7]-m[0] * m[9] * m[10] * m[7]-m[8] * m[1] * m[14] * m[7]+m[0] * m[5] * m[14] * m[7]+
    m[12] * m[1] * m[2] * m[11]-m[4] * m[9] * m[2] * m[11]-m[12] * m[1] * m[6] * m[11]+m[0] * m[9] * m[6] * m[11]+
    m[4] * m[1] * m[14] * m[11]-m[0] * m[1] * m[14] * m[11]-m[8] * m[1] * m[2] * m[15]+m[4] * m[5] * m[2] * m[15]+
    m[8] * m[1] * m[6] * m[15]-m[0] * m[5] * m[6] * m[15]-m[4] * m[1] * m[10] * m[15]+m[0] * m[1] * m[10] * m[15];
  return value;
  */
}
/* --------------------------------------------------------------------------
   look at
   -------------------------------------------------------------------------- */ 
void matrix_lookAtf(Mat4x4f m, Vector3f *pos, Vector3f *lookAt, Vector3f *up){

  Vector3f viewdir; 
  Vector3f upn; 
  viewdir.x = lookAt->x - pos->x;
  viewdir.y = lookAt->y - pos->y;
  viewdir.z = lookAt->z - pos->z;
  
  vector_normalize3f(&viewdir,&viewdir);
  vector_normalize3f(&upn,up);
  
  Vector3f s,u;

  vector_crossProd3f(&s,&viewdir,&upn);
  vector_crossProd3f(&u,&s,&viewdir);

  Mat4x4f r;
  matrix_identity4x4f(r);
  
  r[0] = s.x;        r[4] = s.y;        r[ 8] = s.z;        //r[12] = XXX
  r[1] = u.x;        r[5] = u.y;        r[ 9] = u.z;        //r[13] = XXX
  r[2] = -viewdir.x; r[6] = -viewdir.y; r[10] = -viewdir.z; //r[14] = XXX
  //r[3] = XXX       r[7] = XXX         r[11] = XXX         //r[15] = XXX
  

  matrix_translate4x4f(r,-pos->x,-pos->y,-pos->z);  
  
  memcpy(m,r,sizeof(Mat4x4f));

}

/* --------------------------------------------------------------------------
   Inversion
   -------------------------------------------------------------------------- */ 
int matrix_invert4x4f(Mat4x4f r, Mat4x4f m) {
  
  Mat4x4f tmp;


  float a0 = m[ 0]*m[ 5] - m[ 1]*m[ 4];
  float a1 = m[ 0]*m[ 6] - m[ 2]*m[ 4];
  float a2 = m[ 0]*m[ 7] - m[ 3]*m[ 4];
  float a3 = m[ 1]*m[ 6] - m[ 2]*m[ 5];
  float a4 = m[ 1]*m[ 7] - m[ 3]*m[ 5];
  float a5 = m[ 2]*m[ 7] - m[ 3]*m[ 6];
  float b0 = m[ 8]*m[13] - m[ 9]*m[12];
  float b1 = m[ 8]*m[14] - m[10]*m[12];
  float b2 = m[ 8]*m[15] - m[11]*m[12];
  float b3 = m[ 9]*m[14] - m[10]*m[13];
  float b4 = m[ 9]*m[15] - m[11]*m[13];
  float b5 = m[10]*m[15] - m[11]*m[14];

  float det = a0*b5 - a1*b4 + a2*b3 + a3*b2 - a4*b1 + a5*b0;
  if (det != 0.0)
    {

      tmp[ 0] = + m[ 5]*b5 - m[ 6]*b4 + m[ 7]*b3;
      tmp[ 4] = - m[ 4]*b5 + m[ 6]*b2 - m[ 7]*b1;
      tmp[ 8] = + m[ 4]*b4 - m[ 5]*b2 + m[ 7]*b0;
      tmp[12] = - m[ 4]*b3 + m[ 5]*b1 - m[ 6]*b0;
      tmp[ 1] = - m[ 1]*b5 + m[ 2]*b4 - m[ 3]*b3;
      tmp[ 5] = + m[ 0]*b5 - m[ 2]*b2 + m[ 3]*b1;
      tmp[ 9] = - m[ 0]*b4 + m[ 1]*b2 - m[ 3]*b0;
      tmp[13] = + m[ 0]*b3 - m[ 1]*b1 + m[ 2]*b0;
      tmp[ 2] = + m[13]*a5 - m[14]*a4 + m[15]*a3;
      tmp[ 6] = - m[12]*a5 + m[14]*a2 - m[15]*a1;
      tmp[10] = + m[12]*a4 - m[13]*a2 + m[15]*a0;
      tmp[14] = - m[12]*a3 + m[13]*a1 - m[14]*a0;
      tmp[ 3] = - m[ 9]*a5 + m[10]*a4 - m[11]*a3;
      tmp[ 7] = + m[ 8]*a5 - m[10]*a2 + m[11]*a1;
      tmp[11] = - m[ 8]*a4 + m[ 9]*a2 - m[11]*a0;
      tmp[15] = + m[ 8]*a3 - m[ 9]*a1 + m[10]*a0;

      float inv = 1.0f/det;
      tmp[ 0] *= inv;
      tmp[ 1] *= inv;
      tmp[ 2] *= inv;
      tmp[ 3] *= inv;
      tmp[ 4] *= inv;
      tmp[ 5] *= inv;
      tmp[ 6] *= inv;
      tmp[ 7] *= inv;
      tmp[ 8] *= inv;
      tmp[ 9] *= inv;
      tmp[10] *= inv;
      tmp[11] *= inv;
      tmp[12] *= inv;
      tmp[13] *= inv;
      tmp[14] *= inv;
      tmp[15] *= inv;

      for (int i = 0; i < 16; ++i){
	r[i] = tmp[i];
      }
	
      return 0;
    }
    

  return -1;
}


void matrix_toString4x4f(char *str, unsigned int n, Mat4x4f m) {
  
  snprintf(str, n, 
	  "%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f", 
	  m[0],m[4],m[ 8],m[12], 
	  m[1],m[5],m[ 9],m[13],
	  m[2],m[6],m[10],m[14],
	  m[3],m[7],m[11],m[15]);

}

void matrix_transpose4x4f(Mat4x4f r, Mat4x4f m) {
  r[0] = m[0]; 
  r[1] = m[4];
  r[2] = m[8]; 
  r[3] = m[12];
  r[4] = m[1]; 
  r[5] = m[5]; 
  r[6] = m[9]; 
  r[7] = m[13];
  r[8] = m[2]; 
  r[9] = m[6]; 
  r[10] = m[10]; 
  r[11] = m[14];
  r[12] = m[3];
  r[13] = m[7];
  r[14] = m[11]; 
  r[15] = m[15];

}


/* --------------------------------------------------------------------------
   3x3 matrix stuff
   -------------------------------------------------------------------------- */ 

void matrix_sub3x3f(Mat3x3f r, Mat4x4f m) {

  r[0] = m[ 0];
  r[1] = m[ 1]; 
  r[2] = m[ 2];
  
  r[3] = m[ 4];
  r[4] = m[ 5];
  r[5] = m[ 6];
  
  r[6] = m[ 8];
  r[7] = m[ 9];
  r[8] = m[10];

}

void matrix_identity3x3f(Mat3x3f m) {
  memset(m,0,sizeof(float) * 9); 
  m[0] = m[4] = m[8] = 1.0f;
}
void matrix_zero3x3f(Mat3x3f m){
  memset(m,0,sizeof(float) * 9); 
}

void matrix_mul3x3f(Mat3x3f r, const  Mat3x3f a, const Mat3x3f b) {

  int i;

  Mat4x4f tmp;

  for (i = 0; i < 9; i++) {
    tmp[i] = 0.0f;
    
    for(int k = 0; k < 3; k++) {
      tmp[i] += a[(i%3)+k*3] * b[k+(i/3)*3];
    }
  }
  
  for (i = 0; i < 9; i++) {
    r[i] = tmp[i];
  }
}


float matrix_det3x3f(Mat3x3f m) {

  float value =
    m[0]*m[4]*m[8] + m[3]*m[7]*m[2] + m[6]*m[1]*m[5] - 
    m[0]*m[4]*m[5] - m[3]*m[1]*m[8] - m[6]*m[4]*m[2];
  return value;
 
}

void matrix_transpose3x3f(Mat3x3f r, Mat3x3f m){

  Mat3x3f tmp; 

  tmp[0] = m[0];
  tmp[1] = m[3];
  tmp[2] = m[6];
  tmp[3] = m[1];
  tmp[4] = m[4];
  tmp[5] = m[7];
  tmp[6] = m[2];
  tmp[7] = m[5];
  tmp[8] = m[8]; 
  

  for (int i =0; i < 9; ++i) {
    r[i] = tmp[i];
  }

}

void matrix_adjunct3x3f(Mat3x3f r, Mat3x3f m) {

  Mat3x3f tmp;
  
  tmp[0]=         m[4]*m[8]-(m[5]*m[7]);
  tmp[3]=(-1.0f)*(m[1]*m[8]-(m[2]*m[7]));
  tmp[6]=         m[1]*m[5]-(m[2]*m[4]);
  
  
  
  tmp[1]=(-1.0f)*(m[3]*m[8]-m[5]*m[6]);
  tmp[4]=         m[0]*m[8]-m[2]*m[6];
  tmp[7]=(-1.0f)*(m[0]*m[2]-m[2]*m[3]);
  
  
  tmp[2]=         m[3]*m[7]-m[4]*m[6];
  tmp[5]=(-1.0f)*(m[0]*m[7]-m[1]*m[6]);
  tmp[8]=         m[0]*m[4]-m[1]*m[3];


  for (int i =0; i < 9; ++i) {
    r[i] = tmp[i];
  }
}

int  matrix_invert3x3f(Mat3x3f r, Mat3x3f m) {
  
    Mat3x3f tmp;

    // Compute the adjoint.
    tmp[0] = m[4]*m[8] - m[5]*m[7];
    tmp[1] = m[2]*m[7] - m[1]*m[8];
    tmp[2] = m[1]*m[5] - m[2]*m[4];
    tmp[3] = m[5]*m[6] - m[3]*m[8];
    tmp[4] = m[0]*m[8] - m[2]*m[6];
    tmp[5] = m[2]*m[3] - m[0]*m[5];
    tmp[6] = m[3]*m[7] - m[4]*m[6];
    tmp[7] = m[1]*m[6] - m[0]*m[7];
    tmp[8] = m[0]*m[4] - m[1]*m[3];

    float det = m[0]*tmp[0] + m[1]*tmp[3] + m[2]*tmp[6];

    if (det != 0.0){
        float invDet = 1.0/det;
        tmp[0] *= invDet;
        tmp[1] *= invDet;
        tmp[2] *= invDet;
        tmp[3] *= invDet;
        tmp[4] *= invDet;
        tmp[5] *= invDet;
        tmp[6] *= invDet;
        tmp[7] *= invDet;
        tmp[8] *= invDet;
        
	
	for (int i = 0; i < 9; ++i) {
	  r[i] = tmp[i];
	}
	return 0;
    }

    return -1;

  

}


void matrix_vecMul3x3f(Vector3f r, Mat3x3f m, Vector3f v){

}
