


#ifndef __MESH_H
#define __MESH_H

#include "vector.h"


typedef struct{ 
  
  Vector3f *vertices;
  Vector3f *normals;
  Vector3f *colors;
  Vector2f *texCoords;
  
  int      *faces[3];
} Mesh;  


  



#endif
  
