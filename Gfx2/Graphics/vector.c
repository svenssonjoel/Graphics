
#include "vector.h" 
Vector2f vector2f(float x, float y){
  Vector2f r; 
  r.x = x; 
  r.y = y; 
  return r;
}

Vector3f vector3f(float x, float y, float z) {
  Vector3f r; 
  r.x = x; 
  r.y = y; 
  r.z = z; 
  return r;
}
Vector4f vector4f(float x, float y, float z, float w){
  Vector4f r; 
  r.x = x; 
  r.y = y; 
  r.z = z; 
  r.w = w; 
  return r;
}

/* ----------------------------------------------------------------------------- 
   DotProd
   -------------------------------------------------------------------------- */
float vector_dotProd2f(Vector2f *a, Vector2f *b) {
  return (a->x * b->x + a->y * b->y);
}

float vector_dotProd3f(Vector3f *a, Vector3f *b) {
  return (a->x * b->x + a->y * b->y + a->z * b->z);
}

float vector_dotProd4f(Vector4f *a, Vector4f *b) {
  return (a->x * b->x + a->y * b->y + a->z * b->z + a->w * b->w);
}

/* ----------------------------------------------------------------------------- 
   Add and Subtract
   -------------------------------------------------------------------------- */

void vector_add2f(Vector2f *r, Vector2f *u, Vector2f *v){
  Vector2f tmp;
  tmp.x = u->x + v->x;
  tmp.y = u->y + v->y;
  r->x = tmp.x;
  r->y = tmp.y;
}

void vector_add3f(Vector3f *r, Vector3f *u, Vector3f *v){
  Vector3f tmp;
  tmp.x = u->x + v->x;
  tmp.y = u->y + v->y;
  tmp.z = u->z + v->z;
  r->x = tmp.x;
  r->y = tmp.y;
  r->z = tmp.z;
}
void vector_add4f(Vector4f *r, Vector4f *u, Vector4f *v){
  Vector4f tmp;
  tmp.x = u->x + v->x;
  tmp.y = u->y + v->y;
  tmp.z = u->z + v->z;
  tmp.w = u->w + v->w;
  r->x = tmp.x;
  r->y = tmp.y;
  r->z = tmp.z;
  r->w = tmp.w;
}

void vector_sub2f(Vector2f *r, Vector2f *u, Vector2f *v){
  Vector2f tmp;
  tmp.x = u->x - v->x;
  tmp.y = u->y - v->y;
  r->x = tmp.x;
  r->y = tmp.y;
}
void vector_sub3f(Vector3f *r, Vector3f *u, Vector3f *v){
  Vector3f tmp;
  tmp.x = u->x - v->x;
  tmp.y = u->y - v->y;
  tmp.z = u->z - v->z;
  r->x = tmp.x;
  r->y = tmp.y;
  r->z = tmp.z;
}
void vector_sub4f(Vector4f *r, Vector4f *u, Vector4f *v){
  Vector4f tmp;
  tmp.x = u->x - v->x;
  tmp.y = u->y - v->y;
  tmp.z = u->z - v->z;
  tmp.w = u->w - v->w;
  r->x = tmp.x;
  r->y = tmp.y;
  r->z = tmp.z;
  r->w = tmp.w;
}



/* ----------------------------------------------------------------------------- 
   crossProd3
   -------------------------------------------------------------------------- */
void vector_crossProd3f(Vector3f *result, Vector3f *u, Vector3f *v){
  
  result->x = (u->y * v->z) - (u->z * v->y);
  result->y = (u->z * v->x) - (u->x * v->z);
  result->z = (u->x * v->y) - (u->y * v->x);

}

/* ----------------------------------------------------------------------------- 
   normalize
   -------------------------------------------------------------------------- */
void vector_normalize2f(Vector2f *result,Vector2f *v){
  
  float l = sqrt(v->x*v->x + v->y*v->y); 
  
  result->x = v->x / l;
  result->y = v->y / l; 
}

void vector_normalize3f(Vector3f *result,Vector3f *v){
  
  float l = sqrt(v->x*v->x + v->y*v->y + v->z*v->z); 
  
  result->x = v->x / l;
  result->y = v->y / l; 
  result->z = v->z / l;
}

void vector_normalize4f(Vector4f *result,Vector4f *v){
  
  float l = sqrt(v->x*v->x + v->y*v->y + v->z*v->z + v->w*v->w); 
  
  result->x = v->x / l;
  result->y = v->y / l; 
  result->z = v->z / l;
  result->w = v->w / l;
}


/* ----------------------------------------------------------------------------- 
   Orthogonality checks 
   -------------------------------------------------------------------------- */
bool vector_isOrthogonal2f(Vector2f *a, Vector2f *b) {
  if (VECTOR_DOTPROD2F(a,b) == 0.0 )
    return true;
  else 
    return false;
}

bool vector_isOrthogonal3f(Vector3f *a, Vector3f *b) {
  if (VECTOR_DOTPROD3F(a,b) == 0.0 )
    return true;
  else 
    return false;
}

bool vector_isOrthogonal4f(Vector4f *a, Vector4f *b) {
  if (VECTOR_DOTPROD4F(a,b) == 0.0 )
    return true;
  else 
    return false;
}


/* ----------------------------------------------------------------------------- 
   genNormal: Given three points returns a normal
   -------------------------------------------------------------------------- */
void vector_genNormal3f(Vector3f *v, Vector3f *p1, Vector3f *p2, Vector3f *p3){ 
  Vector3f a,b;
  
  a.x = p2->x - p1->x;
  a.y = p2->y - p1->y;
  a.z = p2->z - p1->z;


  b.x = p3->x - p1->x;
  b.y = p3->y - p1->y;
  b.z = p3->z - p1->z;

  vector_crossProd3f(v,&a,&b);
  vector_normalize3f(v,v);

}

