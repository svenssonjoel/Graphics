/* frame.c */

#include "frame.h"
#include "vector.h"
#include "matrix.h"

/* --------------------------------------------------------------------------
   Frames
   -------------------------------------------------------------------------- */ 
/* 
   what are frames 

*/ 

/* --------------------------------------------------------------------------
   create
   -------------------------------------------------------------------------- */ 
void frame_create(Frame *f, Vector3f *pos, Vector3f *up, Vector3f *look){
  f->pos.x = pos->x;
  f->pos.y = pos->y;
  f->pos.z = pos->z;
  f->up.x = up->x;
  f->up.y = up->y;
  f->up.z = up->z;
  f->look.x = look->x;
  f->look.y = look->y;
  f->look.z = look->z;
}



void frame_getTransform(Mat4x4f m, Frame *f){
  Vector3f right; 
  
  vector_crossProd3f(&right,&f->up, &f->look); 
  m[0]  = right.x;
  m[1]  = right.y;
  m[2]  = right.z; 
  m[3]  = 0.0f;
  m[4]  = f->up.x;
  m[5]  = f->up.y;
  m[6]  = f->up.z;
  m[7]  = 0.0f;
  m[8]  = f->look.x;
  m[9]  = f->look.y;
  m[10] = f->look.z;
  m[11] = 0.0f;
  m[12] = f->pos.x;
  m[13] = f->pos.y;
  m[14] = f->pos.z;
  m[15] = 1.0f; 
}

void frame_cameraTransform(Mat4x4f m, Frame *f){ 

  Mat4x4f  trans; 
  Vector3f negLook; 
  Vector3f right; 

  negLook.x = -f->look.x;
  negLook.y = -f->look.y;
  negLook.z = -f->look.z; 
  
  vector_crossProd3f(&right,&f->up, &negLook);

  m[0]  = right.x;
  m[4]  = right.y; 
  m[8]  = right.z; 
  m[12] = 0.0f;
  m[1]  = f->up.x;
  m[5]  = f->up.y; 
  m[9]  = f->up.z; 
  m[13] = 0.0f; 
  m[2]  = negLook.x;
  m[6]  = negLook.y; 
  m[10] = negLook.z; 
  m[14] = 0.0f; 
  m[3]  = 0.0f; 
  m[7]  = 0.0f; 
  m[11] = 0.0f; 
  m[15] = 1.0f; 

  matrix_identity4x4f(trans);
  matrix_translate4x4f(trans,-f->pos.x,-f->pos.y,-f->pos.z);
  
  matrix_mul4x4f(m,m,trans); 
}

 
