
#ifndef __VECTOR_H
#define __VECTOR_H

#include <math.h>
#include <stdbool.h>

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

extern Vector2f vector2f(float x, float y);
extern Vector3f vector3f(float x, float y, float z);
extern Vector4f vector4f(float x, float y, float z, float w);

extern float vector_dotProd2f(Vector2f *, Vector2f *);
extern float vector_dotProd3f(Vector3f *, Vector3f *);
extern float vector_dotProd4f(Vector4f *, Vector4f *);

extern void vector_crossProd3f(Vector3f *, Vector3f *, Vector3f *);

extern void vector_add2f(Vector2f *r, Vector2f *u, Vector2f *v);
extern void vector_add3f(Vector3f *r, Vector3f *u, Vector3f *v);
extern void vector_add4f(Vector4f *r, Vector4f *u, Vector4f *v);

extern void vector_sub2f(Vector2f *r, Vector2f *u, Vector2f *v);
extern void vector_sub3f(Vector3f *r, Vector3f *u, Vector3f *v);
extern void vector_sub4f(Vector4f *r, Vector4f *u, Vector4f *v);

extern void vector_crossProd3f(Vector3f *result, Vector3f *u, Vector3f *v);

extern void vector_normalize2f(Vector2f *result, Vector2f *v);
extern void vector_normalize3f(Vector3f *result, Vector3f *v); 
extern void vector_normalize4f(Vector4f *result, Vector4f *v); 

extern bool vector_isOrthogonal2f(Vector2f *a, Vector2f *b);
extern bool vector_isOrthogonal3f(Vector3f *a, Vector3f *b);
extern bool vector_isOrthogonal4f(Vector4f *a, Vector4f *b);

#define VECTOR_DOTPROD2F(a,b) (a->x * b->x + a->y * b->y)
#define VECTOR_DOTPROD3F(a,b) (a->x * b->x + a->y * b->y + a->z * b->z)
#define VECTOR_DOTPROD4F(a,b) (a->x * b->x + a->y * b->y + a->z * b->z + a->w * b->w)

#define VECTOR_LENGTH2F(v) (sqrt(v->x*v->x + v->y*v->y))
#define VECTOR_LENGTH3F(v) (sqrt(v->x*v->x + v->y*v->y + v->z*v->z))
#define VECTOR_LENGTH4F(v) (sqrt(v->x*v->x + v->y*v->y + v->z*v->z + v->w*v->w))

extern void vector_genNormal3f(Vector3f *v, Vector3f *p1, Vector3f *p2, Vector3f *p3);

/* find out when it is risky to do this */
#define VECTOR_SUB3F(V)  (Vector3f*)V
#define VECTOR_SUB2F(V)  (Vector2f*)V

#endif
