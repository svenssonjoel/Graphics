/* matrix.h */

#ifndef __MATRIX_H
#define __MATRIX_H

#include "vector.h"

typedef float  Mat3x3f[9];
typedef double Mat3x3d[9];
typedef float  Mat4x4f[16];
typedef double Mat4x4d[16];


/* 4x4 float matrices */ 
extern void matrix_identity4x4f(Mat4x4f);
extern void matrix_mul4x4f(Mat4x4f r, const  Mat4x4f a, const Mat4x4f b);

extern void matrix_zero4x4f(Mat4x4f input);
extern void matrix_translate4x4f(Mat4x4f r, float x, float y, float z);
extern void matrix_scale4x4f(Mat4x4f r, float x, float y, float z);
extern void matrix_rotate4x4f(Mat4x4f r,float rad, float x,float y, float z);

extern float matrix_det4x4f(Mat4x4f m);

extern void matrix_orthof(Mat4x4f m, float l, float r, float b, float t, float n, float f);
extern void matrix_ortho2Df(Mat4x4f m, float l, float r, float b, float t);

extern void matrix_frustumf(Mat4x4f m, float l, float r, float b, float t, float n, float f);
extern void matrix_lookAtf(Mat4x4f m, Vector3f *pos, Vector3f *lookAt, Vector3f *up);

extern int  matrix_invert4x4f(Mat4x4f r, Mat4x4f m);
extern void matrix_toString4x4f(char *str, unsigned int n, Mat4x4f m);

extern void matrix_transpose4x4f(Mat4x4f r, Mat4x4f m);

extern void matrix_transform4x4f(Vector4f *r, Mat4x4f m, Vector4f *v);

/* 3x3 float matrices */
extern void matrix_sub3x3f(Mat3x3f r, Mat4x4f m);

extern void matrix_identity3x3f(Mat3x3f m);
extern void matrix_zero3x3f(Mat3x3f m);
extern void matrix_mul3x3f(Mat3x3f r, const  Mat3x3f a, const Mat3x3f b);
extern float matrix_det3x3f(Mat3x3f m);
extern void matrix_transpose3x3f(Mat3x3f r, Mat3x3f m);
extern int matrix_invert3x3f(Mat3x3f r, Mat3x3f m);

extern void matrix_vecMul3x3f(Vector3f r, Mat3x3f m, Vector3f v);

#endif
