


#ifndef __SIMPLEMESH_H
#define __SIMPLEMESH_H


#include <stdbool.h>

#include "vector.h"
#include "matrix.h"
#include "shader.h"


/* 
   simplemesh.h 
    - simplify and "make more clear the scope of" what was originially in file mesh.h
    
   Whats a mesh, 
    A bunch of triangles that in some way "belong" together.
      
*/


/* -----------------------------------------------------------------------------
   MESH STRUCTURE
   -------------------------------------------------------------------------- */
typedef struct{ 
  
  Vector3f *vertices;        
  Vector3f *normals;
  Vector3f *colors;
  Vector2f *texels;
  unsigned int num_vertices;

  // TODO: learn about vaos, what are they and why and so on.
  GLuint vao;  // Vertex Array Object
  GLuint vertex_buffer; 
  GLuint normal_buffer;
  GLuint color_buffer;
  GLuint texel_buffer;
  GLuint index_buffer;
  
  // Removing texture_id and textured
  // from mesh. If a mesh has texels you can render 
  // it with any suitable texture.
  //GLuint  texture_id;
  //bool   textured;

  unsigned char *indices; //the values here can be of type 
                          //different from "unsigned char"
  unsigned int num_indices;
  
  GLenum   indices_type; // GL_UNSIGNED_BYTE,
                         // GL_UNSIGNED_SHORT, 
                         // GL_UNSIGNED_INT
                         // and so on 
} Mesh;

//Work in progress 
typedef struct { 
  GLuint vao; 

  GLuint index_buffer; 
  unsigned char *indices; 
  unsigned int   num_indices;
  GLenum         indices_type; 
} CMeshVAO; 

typedef struct { 
  
  GLuint vertex_buffer; 
  GLuint normal_buffer;
  GLuint color_buffer;
  GLuint texel_buffer;
  GLuint index_buffer;

  unsigned char *indices; 
  unsigned int num_indices;
  
  GLenum   indices_type; 
} CMeshNoVAO; 
   

extern Mesh *mesh_create(); // initiate all fields to ZERO, 0, NULL
extern Mesh *mesh_init(Mesh *m);

/* Allocate space for attributes on CPU side (vertices,normals, texels) */
extern void mesh_allocateAttribs(Mesh *m, unsigned int, unsigned long mode);
/* Free the attribute space on the CPU side, if uploaded the attributes will 
   still exist on the GPU side only */ 
extern void mesh_freeAttribs(Mesh *m);

/* Copy all attribute data into a GPU side buffer */ 
extern void mesh_upload(Mesh *m);
extern void mesh_upload_prim(Mesh *m);

/* Destroy completely */
extern void mesh_destroy(Mesh *m);

/*general rendering procedure */
extern void mesh_render(Shader *s, Mesh *m);
extern void mesh_render_prim(Shader *s, Mesh *m);
extern void mesh_renderTex(Shader *s, GLuint textureID, Mesh *m);
extern void mesh_renderTex_prim(Shader *s, GLuint textureID, Mesh *m);


/*specialized rendering procedures */
// TODO: These are old, phase out!
extern void mesh_renderDot(Shader *s, Mesh *m);
extern void mesh_renderFill(Shader *s, Mesh *m);
extern void mesh_renderFillLit(Shader *s, Mesh *m); 
extern void mesh_renderTextured(Shader *s, Mesh *m);
extern void mesh_renderTexturedLit(Shader *s, Mesh *m);


#define MESH_LOAD_NORMALS   1
#define MESH_LOAD_TEXELS    1<<1

#define MESH_VERTICES       1
#define MESH_TEXELS         1<<1
#define MESH_NORMALS        1<<2
#define MESH_COLORS         1<<3


/* creates a mesh and allocates its memory.
   Then fills that memory with the contents of the 3ds file */

extern int mesh_load3ds(Mesh *m, char *filename, unsigned int mode);
#endif
  
