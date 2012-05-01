

#include <lib3ds/file.h>
#include <lib3ds/mesh.h>



int main(void) {

  Lib3dsFile *model = NULL;
  int n_meshes = 0;
  int n_faces  = 0;
  Lib3dsMesh *mesh;
  long n_points = 0;
  long n_texcoords = 0;
  
  printf("info3ds: starting up!\n");

  model = lib3ds_file_load("monkey.3ds");
  
  if (!model) {
    printf("info3ds: error loading file\n");
  }
  
  /* count meshes (a linked list)*/
  for (mesh = model->meshes; mesh != NULL; mesh = mesh->next) {
    n_meshes ++;
    
    n_faces += mesh->faces;
    n_points += mesh->points;
    n_texcoords += mesh->texels;
  }
  
  printf("info3ds: number of meshes %d\n",n_meshes);
  printf("info3ds: number of faces %d\n",n_faces);
  printf("info3ds: number of points %ld\n",n_points);
  printf("info3ds: number of texcoords %ld\n",n_texcoords);

  
}
