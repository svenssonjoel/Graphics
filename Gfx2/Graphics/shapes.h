
#ifndef _SHAPES_H_ 
#define _SHAPES_H_

#include "vector.h"
#include "image.h"
/* 
   TODO: 
     Add: More shapes.
     Add: Texture coordinates into some of the existing ones.
           
*/

#define SHAPES_TETRAHEDRON_VERTICES 12
#define SHAPES_TETRAHEDRON_NORMALS  12

#define SHAPES_PRISM_VERTICES 24
#define SHAPES_PRISM_NORMALS  24

/* sphere detail to number of vertices */
#define SHAPES_SPHERE_VERTICES(X) (((X+1)*X/2)+(X+1))
#define SHAPES_SPHERE_INDICES(X)  ((X*X/2)*6)

extern void shapes_quad(Vector3f *v_data, 
			Vector3f *n_data, 
			Vector2f *t_data, 
			unsigned char *i_data,
			float w, float h);

void shapes_disc(Vector3f *v_data, 
		 Vector3f *n_data, 
		 Vector2f *t_data, 
		 unsigned short *i_data,
		 int n,
		 float r);

extern void shapes_tetrahedron(Vector3f *v_data, Vector3f *n_data, float k, float h);
extern void shapes_prism(Vector3f *v_data, Vector3f *n_data, float w, float h, float d);


extern void shapes_flatMesh(Vector3f *v_data, 
			    Vector3f *n_data, 
			    Vector2f *t_data,
			    unsigned int *i_data,
			    int w, int d, float xstep,float zstep,
			    int texture_repeat);

extern void shapes_randMesh(Vector3f *v_data, Vector3f *n_data, unsigned int *i_data,
			    int w, int d, float xstep,float zstep);


extern void shapes_meshFromImg(Vector3f *v_data, 
			       Vector3f *n_data,
			       Vector2f *t_data, 
			       unsigned int *i_data,
			       int w, int d, float xstep,float zstep, Image *img);


/* ----------------------------------------------------------------------------
   Surfaces of revolution
   ------------------------------------------------------------------------- */


extern void shapes_cylinder(Vector3f *v_data, 
			    Vector3f *n_data,
			    Vector2f *t_data, 
			    unsigned int *i_data,
			    int r, int h, float rsteps,float hsteps);

  
extern void shapes_sphere(Vector3f *v_data, 
			  Vector3f *n_data,
			  Vector2f *t_data, 
			  unsigned int *i_data,
			  float r, int rsteps);
 
#endif
