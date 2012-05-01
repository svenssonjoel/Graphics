

#include "multimesh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 

#include <lib3ds/file.h>
#include <lib3ds/mesh.h>

int multimesh_load3ds(Multimesh *mm, char* filename, unsigned int mode) {
  
  Lib3dsFile *model = NULL;
  int n_meshes = 0;
  int n_faces  = 0;
  Lib3dsMesh *mesh;
  int n_points = 0;
  //int n_texels = 0;

  model = lib3ds_file_load(filename);
  mesh  = model->meshes; 


  /* count meshes */
  for (mesh = model->meshes; mesh != NULL; mesh = mesh->next) 
    n_meshes ++;

  mm->meshes = (Mesh*)malloc(n_meshes * sizeof(Mesh));
  memset(mm->meshes,0,sizeof(Mesh)*n_meshes);
  mm->num_meshes = n_meshes;
  
  
  int mi = 0; 
  for (mesh = model->meshes; mesh != NULL; mesh = mesh->next) {

    n_faces = mesh->faces;

    n_points = n_faces * 3;

    mm->meshes[mi].vertices = (Vector3f *)malloc(n_points*sizeof(Vector3f));

    if (mode | MESH_LOAD_TEXELS) 
      mm->meshes[mi].texels = (Vector2f *)malloc(n_points*sizeof(Vector2f));  
    if (mode | MESH_LOAD_NORMALS) 
      mm->meshes[mi].normals = (Vector3f *)malloc(n_points*sizeof(Vector3f));
    
    mm->meshes[mi].num_vertices = n_points;
    mm->meshes[mi].indices = (unsigned char*)malloc(n_points*sizeof(unsigned int));
    mm->meshes[mi].num_indices = n_points;
    mm->meshes[mi].indices_type = GL_UNSIGNED_INT;

    
    int index = 0; 
    for (int i = 0; i < n_faces; ++i) {
      mm->meshes[mi].vertices[index].x   = mesh->pointL[mesh->faceL[i].points[0]].pos[0]; 
      mm->meshes[mi].vertices[index].y   = mesh->pointL[mesh->faceL[i].points[0]].pos[1]; 
      mm->meshes[mi].vertices[index++].z = mesh->pointL[mesh->faceL[i].points[0]].pos[2]; 
      mm->meshes[mi].vertices[index].x   = mesh->pointL[mesh->faceL[i].points[1]].pos[0]; 
      mm->meshes[mi].vertices[index].y   = mesh->pointL[mesh->faceL[i].points[1]].pos[1]; 
      mm->meshes[mi].vertices[index++].z = mesh->pointL[mesh->faceL[i].points[1]].pos[2]; 
      mm->meshes[mi].vertices[index].x   = mesh->pointL[mesh->faceL[i].points[2]].pos[0]; 
      mm->meshes[mi].vertices[index].y   = mesh->pointL[mesh->faceL[i].points[2]].pos[1]; 
      mm->meshes[mi].vertices[index++].z = mesh->pointL[mesh->faceL[i].points[2]].pos[2]; 
    }
    unsigned int * inds = (unsigned int *)mm->meshes[mi].indices;
    index = 0; 
    for (int i = 0; i < n_faces; i ++) {
      for (int j = 0; j < 3; j ++) {
	
	inds[index] = index; // (unsigned int)mesh->faceL[i].points[j];
	//printf("%d", inds[index]);
	index ++;
      }
    }
    
    Lib3dsVector *n = malloc(n_points * sizeof(Lib3dsVector));
  

    lib3ds_mesh_calculate_normals(mesh,n);
    
    //memcpy(m->normals,n,n_points*sizeof(Vector3f));
    for (int i = 0; i < n_points; i ++) {
      mm->meshes[mi].normals[i].x = n[i][0];
      mm->meshes[mi].normals[i].y = n[i][1];
      mm->meshes[mi].normals[i].z = n[i][2];
    }
    
    mi++;
  }
  
  
  return 0;
}


void multimesh_renderFillLit(Shader *s,Multimesh *mm){
 
  for (int i = 0; i < mm->num_meshes; ++i) {
    mesh_renderFillLit(s,&mm->meshes[i]);
  }
    
}

void multimesh_renderFill(Shader *s,Multimesh *mm){
 
  for (int i = 0; i < mm->num_meshes; ++i) {
    mesh_renderFill(s,&mm->meshes[i]);
  }
    
}

void multimesh_upload(Multimesh *mm) {
  for (int i = 0; i < mm->num_meshes; ++i) {
    printf("uploading: %d\n",i);
    mesh_upload(&mm->meshes[i]);
   }
}
