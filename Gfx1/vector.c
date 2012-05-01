
#include "vector.h" 


float dotp(Vector3f *a, Vector3f *b) {
  return (a->x * b->x + a->y * b->y + a->z * b->z);
}
