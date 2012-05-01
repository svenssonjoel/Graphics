
#include "shapes.h"
#include "image.h"

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

/*
  TODO: 
     #Calculate normals for the meshes 
     #Find some sound way to give texture coords for:
        Prism
        Tetrahedron
     #Add more shapes

     #SHAPES CYLINDERS AND SPHERES, one "row" of vertices needs to be duplicated
      given different texture coordinates. If not, where the sphere 
      is "glued" together there will be a strange seam. The entire texture 
      will appear again in that thin seam of the sphere. 
      # DONE FOR SPHERE ! (LOOKS OK)
*/


/* ----------------------------------------------------------------------------
   QUAD
   ------------------------------------------------------------------------- */
void shapes_quad(Vector3f *v_data, 
		 Vector3f *n_data, 
		 Vector2f *t_data, 
		 unsigned char *i_data,
		 float w, float h){

  Vector3f vertices[4] = 
    {{ -w, -h, 0.0f},
     {  w, -h, 0.0f},
     {  w,  h, 0.0f},
     { -w,  h, 0.0f}};
  
  Vector3f normals[4] = 
    {{ 0.0f, 0.0f, 1.0f},
     { 0.0f, 0.0f, 1.0f},
     { 0.0f, 0.0f, 1.0f},
     { 0.0f, 0.0f, 1.0f}};
  
  Vector2f texcoords[4] = 
    {{  0.0f, 0.0f},
     {  1.0f, 0.0f},
     {  1.0f, 1.0f},
     {  0.0f, 1.0f}};

    
  unsigned char indices[6] = 
    {0,1,2,   //triangle1
     2,3,0};  //triangle2

  if (v_data) 
    memcpy(v_data,vertices,4*sizeof(Vector3f));
  if (n_data) 
    memcpy(n_data,normals,4*sizeof(Vector3f));
  if (t_data) 
    memcpy(t_data,texcoords,4*sizeof(Vector2f));
  if (i_data) 
    memcpy(i_data,indices,6*sizeof(unsigned char));

}

/* ----------------------------------------------------------------------------
   DISC
   ------------------------------------------------------------------------- */
void shapes_disc(Vector3f *v_data, 
		 Vector3f *n_data, 
		 Vector2f *t_data, 
		 unsigned short *i_data,
		 int n,
		 float r) {
  
  
  //  float x,y; 
  float d = 2.0f*3.14159f / (float)n;
  float t = 0.0f;
  
  /* vertices */
  if (v_data) {
    v_data[0].x = 0.0f;  // the centerpoint
    v_data[0].y = 0.0f; 
    v_data[0].z = 0.0f;
   
    for (int i = 1; i < n+1; i ++) {
      v_data[i].x = r * cos(t);
      v_data[i].y = r * sin(t);
      v_data[i].z = 0.0f;
      t -= d;
    }
  }

  /* normals */
  if (n_data) {
    for (int i=0; i < n+1; i ++) {
      n_data[0].x = 0.0f;
      n_data[0].y = 0.0f;
      n_data[0].z = 1.0f;
    }
  }
  
  /* texture coords 
     Given by the unit circle translated so that its center is at (0.5,0.5)*/
  t = 0.0f;
  if (t_data) {
    t_data[0].x = 0.5f;
    t_data[0].y = 0.5f;
    
    for (int i = 1; i < n+1; i++) {
      t_data[i].x = (1.0f + cos(t))/2.0f;
      t_data[i].y = (1.0f + sin(t))/2.0f;
      t -= d;
    }
  }
  
  /* indices */ 
  if (i_data) {
    int index = 0;
    for (int i = 0; i < n; i ++ ) {
      i_data[index++] = i+1;
      i_data[index++] = 0;
      i_data[index++] = (i+1) % n + 1;
    }

  }
  
  
  
}

/* ----------------------------------------------------------------------------
   Tetrahedron
   ------------------------------------------------------------------------- */
void shapes_tetrahedron(Vector3f *v_data, Vector3f *n_data, float k, float h) {
  
  float a = sqrt ((k * k) / 2.0);
  float h2 = h / 2.0f;
    
  Vector3f vertices[12] = 
    {{   a,  -a,  h2},
     { 0.0, 0.0, -h2},
     { 0.0,   k,  h2}, 
     
     {   a,  -a,  h2},
     { 0.0,   k,  h2}, 
     {  -a,  -a,  h2},

     {  -a,  -a,  h2}, 
     { 0.0,   k,  h2},
     { 0.0, 0.0, -h2},
    
     {   a,  -a,  h2},
     {  -a,  -a,  h2},
     { 0.0, 0.0, -h2}};
  if (v_data) {
    memcpy(v_data,vertices,12*sizeof(Vector3f));
  }
  Vector3f normal;
  if (n_data) {
    for (int i = 0; i < 4; ++i) {
      vector_genNormal3f(&normal,
		&vertices[i*3],
		&vertices[i*3+1],
		&vertices[i*3+2]);
      n_data[i*3].x = normal.x;
      n_data[i*3].y = normal.y;
      n_data[i*3].z = normal.z;
      
      n_data[i*3+1].x = normal.x;
      n_data[i*3+1].y = normal.y;
      n_data[i*3+1].z = normal.z;

      n_data[i*3+2].x = normal.x;
      n_data[i*3+2].y = normal.y;
      n_data[i*3+2].z = normal.z;
    }
	        
  }
  
}

/* ----------------------------------------------------------------------------
   PRISM
   ------------------------------------------------------------------------- */
void shapes_prism(Vector3f *v_data, Vector3f *n_data, float w, float h, float d){


  float whalf = w / 2.0f;   
  float hhalf = h / 2.0f;
  float dhalf = d / 2.0f;

 
 
  Vector3f vertices[24] =
    {    { whalf,     0,-hhalf}, //0 top 1 
	 {-whalf,     0,-hhalf},
	 { whalf, dhalf, hhalf},   

	 { whalf, dhalf, hhalf}, //1 top 2  
	 {-whalf,     0,-hhalf},
	 {-whalf, dhalf, hhalf},

	 { whalf,-dhalf, hhalf}, //2 sida
	 { whalf,     0,-hhalf},   
	 { whalf, dhalf, hhalf},

	 { whalf,-dhalf, hhalf}, //3 baksida
	 { whalf, dhalf, hhalf},
	 {-whalf, dhalf, hhalf},   

	 { whalf,-dhalf, hhalf}, //4 baksida
	 {-whalf, dhalf, hhalf},
	 {-whalf,-dhalf, hhalf},

	 { whalf,-dhalf, hhalf}, //5 botten
	 {-whalf,-dhalf, hhalf},
	 {-whalf,     0,-hhalf},

	 {-whalf,     0,-hhalf}, //6 botten
	 { whalf,     0,-hhalf},       
	 { whalf,-dhalf, hhalf},

	 {-whalf,-dhalf, hhalf}, //7 sida
	 {-whalf, dhalf, hhalf},   
	 {-whalf,     0,-hhalf}, }; 

  if(v_data)  
    memcpy(v_data,vertices,SHAPES_PRISM_VERTICES*sizeof(Vector3f));

  Vector3f normal;

  if(n_data) {
    for (int i = 0; i < 8; ++i) {
      vector_genNormal3f(&normal,
		&vertices[i*3],
		&vertices[i*3+1],
		&vertices[i*3+2]);
      n_data[i*3].x = normal.x;
      n_data[i*3].y = normal.y;
      n_data[i*3].z = normal.z;
      
      n_data[i*3+1].x = normal.x;
      n_data[i*3+1].y = normal.y;
      n_data[i*3+1].z = normal.z;
      
      n_data[i*3+2].x = normal.x;
      n_data[i*3+2].y = normal.y;
      n_data[i*3+2].z = normal.z;
    }
    
    }
  
}

/* ----------------------------------------------------------------------------
   FLATMESH
   ------------------------------------------------------------------------- */
void shapes_flatMesh(Vector3f *v_data, 
		     Vector3f *n_data, 
		     Vector2f *t_data, 
		     unsigned int *i_data,
		     int w, int d, float xstep,float zstep,
		     int texture_repeat) {

  /* generate a flat mesh consisting of w*d vertices */

  int whalf = w / 2;
  int dhalf = d / 2; 
  
  float dx = xstep; 
  float dz = -zstep;
  
  float x = (float)-whalf; 
  float y = 0; 
  float z = (float)dhalf; 

  float u = 0;
  float v = 0; 
  float du = texture_repeat ? 1.0f : 1.0f / w;
  float dv = texture_repeat ? 1.0f : 1.0f / d;
  
  int index = 0;
  for (int i = 0; i < w; ++i) {
    for (int j = 0; j < d; ++j) {
      
      
      v_data[index].x = x;
      v_data[index].y = y; 
      v_data[index].z = z;

      //if (t_data) {
	t_data[index].x = u;
	t_data[index].y = v; 
	//}
      
      x += dx;
      u += du;
      index++;
    }
    z += dz;
    v += dv;
    u = 0.0f;
    x = (float)-whalf;
  }

  index = 0; 
  for (int j = 0; j < d-1; ++j) {  
    for (int i = 0; i < w-1; ++i) {
      int e = j*w + i; 
      int e1 = j*w + i + 1;
      int e2 = (j+1)*w + i;
      int e3 = (j+1)*w + i + 1;
      
      i_data[index++] = e; // Triangle a
      i_data[index++] = e1;
      i_data[index++] = e2;  
      
      i_data[index++] = e1; // Triangle b
      i_data[index++] = e3;
      i_data[index++] = e2;
     
    }
  }


}

/* ----------------------------------------------------------------------------
   RANDOMMESH
   ------------------------------------------------------------------------- */
void shapes_randMesh(Vector3f *v_data, Vector3f *n_data, unsigned int *i_data,
		     int w, int d, float xstep,float zstep) {

  
  int whalf = w / 2;
  int dhalf = d / 2; 
  
  float dx = xstep; 
  float dz = -zstep;
  
  float x = (float)-whalf; 
  float y = 0; 
  float z = (float)dhalf; 
  
  int index = 0;
  for (int i = 0; i < w; ++i) {
    for (int j = 0; j < d; ++j) {
      
      
      v_data[index].x = x;
      v_data[index].y = y; 
      v_data[index].z = z;
      
      x += dx;
      index++;
    }
    z += dz;
    x = (float)-whalf;
  }


  /* Randomize the y coordinate */ 

  for (int i = 0; i < w*d; ++i) {
    int x1 = i % w;
    int y1 = i / w - 1;
    
    int points[5][2] = 
      {{x1 , y1},
       {x1 - 1, y1},
       {x1 + 1, y1},
       {x1 - 1, y1},
       {x1 + 1, y1}};

    float avg = 0.0f;
    
    for (int k = 0; k < 5; ++k){
      if (!(points[k][0] > w || 
	    points[k][1] > d || 
	    points[k][0] < 0 || 
	    points[k][1] < 0 )) 
	avg += v_data[(points[k][0] + (w * points[k][1]))].y;
    }
    avg = avg / 5.0;
    float r = 0.5 - ((float)(rand()%10000)) / 10000.0f;
    v_data[i].y = avg + r;

  }
  

  /* Create triangles */
  index = 0; 
  for (int j = 0; j < d-1; ++j) {  
    for (int i = 0; i < w-1; ++i) {
      int e = j*w + i; 
      int e1 = j*w + i + 1;
      int e2 = (j+1)*w + i;
      int e3 = (j+1)*w + i + 1;
      
      i_data[index++] = e; // Triangle a
      i_data[index++] = e1;
      i_data[index++] = e2;  
      
      i_data[index++] = e1;
      i_data[index++] = e3;
      i_data[index++] = e2;
     
    }
  }
  /* 
     also calculate normals in each vertex.
     This is probably a case where the average 
     of the normals of the faces that meet in 
     in the vertex is ok. 
  */ 
  
}



/* ----------------------------------------------------------------------------
   MESHFROMIMAGE
   ------------------------------------------------------------------------- */
void shapes_meshFromImg(Vector3f *v_data, 
			Vector3f *n_data,
			Vector2f *t_data, 
			unsigned int *i_data,
			 int w, int d, float xstep,float zstep, Image *img) {

  
  int whalf = (float)w / 2.0f;
  int dhalf = (float)d / 2.0f; 
  
  float dx = xstep; 
  float dz = -zstep;
  
  float x = (float)-whalf; 
  // float y = 0; 
  float z = (float)dhalf;
  
  float u = 0;
  float v = 0; 
  float du = 1.0f / w;
  float dv = 1.0f / d;
    
  
  int index = 0;
  for (int j = 0; j < d; ++j) {
    for (int i = 0; i < w; ++i) {
      
      /* vertex data */
      v_data[index].x = x;
      v_data[index].y = img->data[(j*w+i)*3+2];// use red component only 
      v_data[index].z = z;

      /* texture coord data */
      t_data[index].x = u;
      t_data[index].y = v;

      x += dx;
      u += du;
      index++;
    }
    z += dz;
    v += dv;
    u = 0.0f;
    x = (float)-whalf;
  }

  /* Create triangles */
  index = 0; 
  for (int j = 0; j < d-1; ++j) {  
    for (int i = 0; i < w-1; ++i) {
      int e = j*w + i; 
      int e1 = j*w + i + 1;
      int e2 = (j+1)*w + i;
      int e3 = (j+1)*w + i + 1;
      
      i_data[index++] = e; // Triangle a
      i_data[index++] = e1;
      i_data[index++] = e2;  
      
      i_data[index++] = e1; // Triangle b
      i_data[index++] = e3;
      i_data[index++] = e2;
     
    }
  }
  /* 
     also calculate normals in each vertex.
     This is probably a case where the average 
     of the normals of the faces that meet in 
     in the vertex is ok. 
  */ 
  
}





/* ----------------------------------------------------------------------------
   Surfaces of revolution
   ------------------------------------------------------------------------- */


void shapes_cylinder(Vector3f *v_data, 
		     Vector3f *n_data,
		     Vector2f *t_data, 
		     unsigned int *i_data,
		     int r, int h, float rsteps,float hsteps) {

  
  /* 
     a cylinder generated as surface of revolution, 
     the function to rotate is f(t) = (t,r)
  */
  
  float t = 0.0;
  float dt = (float)h / hsteps; 
  
  float curr_r = 0.0;
  float dr = (2.0*3.14159)/rsteps;
  
  int index = 0; 
  
  for (int i = 0; i < hsteps; i++){
    for (int j = 0; j < rsteps; j ++){
      v_data[index].x = t;
      v_data[index].y = r*cos(curr_r);
      v_data[index].z = r*sin(curr_r);
      
      index++;
      curr_r += dr;
    }
    t += dt;
  }
  

  /* Calculate the texture coords and the normals as well */


}


void shapes_sphere(Vector3f *v_data, 
		   Vector3f *n_data,
		   Vector2f *t_data, 
		   unsigned int *i_data,
		   float r, int rsteps) {
 
  float curr_psi = 0.0;
  float curr_phi = 0.0;
  float dr = (2*3.14159)/rsteps;
  
  int index = 0; 
  
  if (v_data) {
    for (int i = 0; i <= rsteps; i++){
      for (int j = 0; j < (rsteps/2)+1; j ++){ // rotate a half circle 
	v_data[index].x = r*cos(curr_phi);
	v_data[index].y = r*sin(curr_phi)*cos(curr_psi);
	v_data[index].z = r*sin(curr_phi)*sin(curr_psi);
	
	index++;
	curr_phi += dr; //
      }
      curr_phi = 0.0;
      curr_psi += dr;   //
    }
  }
  /* attempting to generate triangles for the surface of the sphere */
  if (i_data) {
    index = 0;
    for (int i = 0; i < rsteps; i++){
      for (int j = 0; j < (rsteps/2); j ++){

	i_data[index++] = i * (rsteps/2+1) + j;
	i_data[index++] = i * (rsteps/2+1) + j+1;
	i_data[index++] = ((i+1) * (rsteps/2+1) + j);
	
	i_data[index++] = i * (rsteps/2+1) + j+1;
	i_data[index++] = (i+1) * (rsteps/2+1) + j+1;
	i_data[index++] = (i+1) * (rsteps/2+1) + j;

      }
    } 
  }

  //printf("Number of Indices computed: %d\n",index);

  /* generation of texture coords for the sphere. 
     Other ways to generate the coords can be imagines. 
     However, this way seems to correspond to a normally 
     occuring way of drawing out planets on a flat surface 
  */
  if (t_data) {
    index = 0; 
    curr_psi = curr_phi = 0.0;
    float dr = (2*3.14159)/(rsteps+1);
    
    for (int i = 0; i <= rsteps; i++){
      for (int j = 0; j < (rsteps/2)+1; j ++){ // rotate a half circle 
	t_data[index].x = curr_psi / (2.0f * 3.14159f);
	t_data[index].y = 1.0f - curr_phi / 3.14159f;

	index++;
	curr_phi += dr; //
      }
      curr_phi = 0.0;
      curr_psi += dr;   //
    }
  }
  
  /* Only calculation of surface normals left */

  
  /* I'm taking the unit sphere, and just considering the 
     points vectors. This should give normals. 
     
     these could of course have been computed in the 
     same loop the actual vertices are computed. 
  */  
  
  if (n_data) {
    index = 0;
    curr_psi = 0.0;
    curr_phi = 0.0;
    for (int i = 0; i <= rsteps; i++){
      for (int j = 0; j < (rsteps/2)+1; j ++){ 
	n_data[index].x = cos(curr_phi);
	n_data[index].y = sin(curr_phi)*cos(curr_psi);
	n_data[index].z = sin(curr_phi)*sin(curr_psi);
	
	index++;
	curr_phi += dr; //
      }
      curr_phi = 0.0;
      curr_psi += dr;   //
    }
  }
}

void shapes_sphere_backup(Vector3f *v_data, 
			  Vector3f *n_data,
			  Vector2f *t_data, 
			  unsigned int *i_data,
			  float r, int rsteps) {
 
  float curr_psi = 0.0;
  float curr_phi = 0.0;
  float dr = (2*3.14159)/rsteps;
  
  int index = 0; 
  
  if (v_data) {
    for (int i = 0; i < rsteps; i++){
      for (int j = 0; j < (rsteps/2)+1; j ++){ // rotate a half circle 
	v_data[index].x = r*cos(curr_phi);
	v_data[index].y = r*sin(curr_phi)*cos(curr_psi);
	v_data[index].z = r*sin(curr_phi)*sin(curr_psi);
	
	index++;
	curr_phi += dr; //
      }
      curr_phi = 0.0;
      curr_psi += dr;   //
    }
  }
  /* attempting to generate triangles for the surface of the sphere */
  if (i_data) {
    index = 0;
    for (int i = 0; i < rsteps; i++){
      for (int j = 0; j < (rsteps/2); j ++){
	
	i_data[index++] = i * (rsteps/2+1) + j;
	i_data[index++] = i * (rsteps/2+1) + j+1;
	i_data[index++] = ((i+1) * (rsteps/2+1) + j) % (rsteps*(rsteps/2+1));
	
	i_data[index++] = i * (rsteps/2+1) + j+1;
	i_data[index++] = ((i+1) * (rsteps/2+1) + j) % (rsteps*(rsteps/2+1)) + 1;
	i_data[index++] = ((i+1) * (rsteps/2+1) + j) % (rsteps*(rsteps/2+1));
	
	
  
      }
    } 
  }

  printf("Number of Indices computed: %d\n",index);

  if (t_data) {
    index = 0; 
    curr_psi = curr_phi = 0.0;
    
    for (int i = 0; i < rsteps; i++){
      for (int j = 0; j < (rsteps/2)+1; j ++){ // rotate a half circle 
	//	t_data[index].x = (1.0 + sin(curr_psi)*cos(curr_phi))/2.0;
	//t_data[index].y = (1.0 + sin(curr_psi)*sin(curr_phi))/2.0;
	t_data[index].x = curr_psi / (2 * 3.14);
	t_data[index].y = curr_phi / 3.14;

	index++;
	curr_phi += dr; //
      }
      curr_phi = 0.0;
      curr_psi += dr;   //
    }
  }
  
    
  

  /*
       u = \sin\theta\cos\phi = \frac{x}{\sqrt{x^2+y^2+z^2}}


    v = \sin\theta\sin\phi = \frac{y}{\sqrt{x^2+y^2+z^2}}    
  */

  

  /* Calculate the texture coords and the normals as well */



 
}
