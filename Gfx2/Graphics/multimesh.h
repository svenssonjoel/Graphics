
#ifndef __MULTIMESH_H_
#define __MULTIMESH_H_


#include "mesh.h"



typedef struct {

  Mesh *meshes; 
  unsigned int num_meshes;

} Multimesh;


extern int multimesh_load3ds(Multimesh *m, char *filename, unsigned int mode); 
extern void multimesh_renderFillLit(Shader *s,Multimesh *mm);
extern void multimesh_renderFill(Shader *s,Multimesh *mm);
extern void multimesh_destroy(Multimesh *m);
extern void multimesh_upload(Multimesh *mm);


#endif 
