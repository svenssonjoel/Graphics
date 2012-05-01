
#ifndef __VECTOR_H
#define __VECTOR_H


typedef struct {
  float x;
  float y;
} Vector2f;

typedef struct {
  float x;
  float y;
  float z;
} Vector3f;

typedef struct {
  float x;
  float y;
  float z;
  float w;
} Vector4f;


float dotp(Vector3f *, Vector3f *);


#endif
