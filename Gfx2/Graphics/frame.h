/* frame.h */

#ifndef __FRAME_H 
#define __FRAME_H 

#include "vector.h"
#include "matrix.h"

typedef struct { 
  Vector3f pos;
  Vector3f up; 
  Vector3f look;
} Frame; 

extern void frame_create(Frame *f, 
			 Vector3f *pos, 
			 Vector3f *up, 
			 Vector3f *look);


extern void frame_cameraTransform(Mat4x4f m, Frame* f);
extern void frame_getTransform(Mat4x4f m, Frame *f);


#endif
