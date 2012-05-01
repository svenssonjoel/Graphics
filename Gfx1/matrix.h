/* matrix.h */

#ifndef __MATRIX_H
#define __MATRIX_H

typedef float Mat4x4[16];

void loadIdentity4x4(Mat4x4);
void mul4x4(Mat4x4 r, const  Mat4x4 a, const Mat4x4 b);
void zero4x4(Mat4x4 input);
void translate4x4(Mat4x4 r, float x, float y, float z);
void scale4x4(Mat4x4 r, float x, float y, float z);
void rotate4x4(Mat4x4 r,float rad, float x,float y, float z);



void setOrtho(Mat4x4 m, float l, float r, float b, float t, float n, float f);
void setOrtho2D(Mat4x4 m, float l, float r, float b, float t);




#endif
