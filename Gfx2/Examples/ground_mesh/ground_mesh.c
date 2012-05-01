// C99 
// OpenGL 3.2
//
// make gl32test

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <GL/glew.h>
#include <GL/freeglut.h> 

#include <math.h>

#include "Graphics/matrix.h"
#include "Graphics/shader.h"
#include "Graphics/image.h"
#include "Graphics/error.h"
#include "Graphics/shapes.h" 
#include "Graphics/texture.h"
#include "Graphics/mesh.h"

#define LEFT 0
#define RIGHT 1
#define UP 2 
#define DOWN 3
int buttons[4]; 

float r = 0.0;
float k = 0.0;

float px = 0.0;  
float pz = 150.0; 
float py = 45.0;

// -----------------------------------------------------------------------------
// Globals.. 
// -----------------------------------------------------------------------------
int win;
int win2;

//GLuint p,f,v;  // program, fragshader, vertexshader

GLuint shader1_frag;
GLuint shader1_vert; 
GLuint shader1_program;

GLuint shader2_frag;
GLuint shader2_vert;  
GLuint shader2_program; 

/* Matrices, Projection, modelview and normal */
Mat4x4f projectionMatrix; 
Mat4x4f modelViewMatrix; 
Mat3x3f normalMatrix;  


GLuint shader1_color_uniloc;
GLuint shader1_vindex;
GLuint shader1_proj;
GLuint shader1_mod; 


GLuint shader2_vindex;
GLuint shader2_tindex;
GLuint shader2_proj;
GLuint shader2_mod;
GLuint shader2_texSampler; 

GLint texture1  = 1;
GLint texture2  = 2;


int window_x = 800; 
int window_y = 600;


Vector3f lightsource = {0, 10, -15}; // POSITION

Image *heightmap;

// -----------------------------------------------------------------------------
// Triangle data
// -----------------------------------------------------------------------------

Vector3f *mesh;
Vector2f *mesh_t;
unsigned int *mesh_index;
GLuint mesh_vert;
GLuint mesh_tex;
GLuint mesh_ix; 
 

Vector3f disc_vertices[101]; 
Vector2f disc_texcoords[101]; 
unsigned short disc_indices[300];
GLuint   disc_vert;
GLuint   disc_tex;
GLuint   disc_ix;



Vector3f cyl_vertices[100]; 
//Vector2f cyl_texcoords[101];  
//unsigned short disc_indices[300];
GLuint   cyl_vert;
//GLuint   disc_tex;
//GLuint   disc_ix;


#define  SPHERE_DETAIL 100
//Vector3f sphere_vertices[SHAPES_SPHERE_VERTICES(SPHERE_DETAIL)]; 
//Vector2f cyl_texcoords[101]; 
//unsigned int *sphere_indices;
GLuint   sphere_vert;
GLuint   sphere_ix;
//GLuint   disc_tex;
//GLuint   disc_ix;

int i;  


Mesh *sphere_mesh;
Uniform sphere_unis[3];
Vector4f mesh_color = {1.0,0.0,0.0,1.0};

Mesh *ground_mesh;
Uniform ground_unis[3]; 
int texunit = 0;

Shader shader1; 
Shader shader2; 


void initBuffer(void) {
  
  //mesh = (Vector3f*)malloc(512*512*sizeof(Vector3f));
  //mesh_t = (Vector2f*)malloc(512*512*sizeof(Vector2f));
  //mesh_index = (unsigned int*)malloc(512*512*6*sizeof(unsigned int));
 
  heightmap = image_loadPNG("Pictures/heightmap1.png");  
  //shapes_randMesh(mesh,NULL,mesh_index,20,20,1.0f,1.0f);
  
  /*  
  glGenBuffers(1, &mesh_vert);
  glBindBuffer(GL_ARRAY_BUFFER, mesh_vert);
  glBufferData(GL_ARRAY_BUFFER, 512*512*sizeof(Vector3f), mesh, GL_STATIC_DRAW);
  printGLError("initBuffer:");
  
  glGenBuffers(1, &mesh_tex);
  glBindBuffer(GL_ARRAY_BUFFER, mesh_tex);
  glBufferData(GL_ARRAY_BUFFER, 512*512*sizeof(Vector2f), mesh_t, GL_STATIC_DRAW);
  printGLError("initBuffer:");
  
  glGenBuffers(1, &mesh_ix);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_ix);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (511*511*6) * sizeof(unsigned int), mesh_index, GL_STATIC_DRAW);
  */
 
  
  ground_mesh->vertices = (Vector3f *)malloc(512*512*sizeof(Vector3f));
  ground_mesh->texels = (Vector2f *)malloc(512*512*sizeof(Vector2f));
  ground_mesh->num_vertices = 512*512;
  ground_mesh->indices = (unsigned char*)malloc(512*512*6*sizeof(unsigned int));
  ground_mesh->num_indices = 512*512*6;
  ground_mesh->indices_type = GL_UNSIGNED_INT;
  ground_mesh->texture_id   = texture1;

  shapes_meshFromImg(ground_mesh->vertices,
		     NULL,
		     ground_mesh->texels,
		     (unsigned int*)ground_mesh->indices,
		     512,512,1.0f,1.0f,heightmap);
  
  mesh_upload(ground_mesh);
  

  /* DISC */ 
  
  shapes_disc(disc_vertices,NULL,disc_texcoords,disc_indices,100,18.0f);
  
  glGenBuffers(1, &disc_vert);
  glBindBuffer(GL_ARRAY_BUFFER, disc_vert);
  glBufferData(GL_ARRAY_BUFFER, 101*sizeof(Vector3f), disc_vertices, GL_STATIC_DRAW);
  printGLError("initBuffer:");

  glGenBuffers(1, &disc_tex);
  glBindBuffer(GL_ARRAY_BUFFER, disc_tex);
  glBufferData(GL_ARRAY_BUFFER, 101*sizeof(Vector3f), disc_texcoords, GL_STATIC_DRAW);
  printGLError("initBuffer:");
  
  glGenBuffers(1, &disc_ix);
  glBindBuffer(GL_ARRAY_BUFFER, disc_ix);
  glBufferData(GL_ARRAY_BUFFER, 300*sizeof(unsigned short), disc_indices, GL_STATIC_DRAW);
  printGLError("initBuffer:");
 
  /* CYLINDER */ 
  
  shapes_cylinder(cyl_vertices,NULL,NULL,NULL,5.0,10.0,10,10);
  
  glGenBuffers(1, &cyl_vert);
  glBindBuffer(GL_ARRAY_BUFFER, cyl_vert);
  glBufferData(GL_ARRAY_BUFFER, 100*sizeof(Vector3f), cyl_vertices, GL_STATIC_DRAW);
  printGLError("initBuffer:");


  /* SPHERE */ 
  //sphere_indices = (unsigned int*)malloc(30000*sizeof(unsigned int));
  sphere_mesh->vertices = (Vector3f *)malloc(SHAPES_SPHERE_VERTICES(SPHERE_DETAIL)*sizeof(Vector3f));
  sphere_mesh->texels = (Vector2f *)malloc(SHAPES_SPHERE_VERTICES(SPHERE_DETAIL)*sizeof(Vector2f));  
  sphere_mesh->texture_id = texture1;
  sphere_mesh->num_vertices = SHAPES_SPHERE_VERTICES(SPHERE_DETAIL);
  sphere_mesh->indices = (unsigned char*)malloc(SHAPES_SPHERE_INDICES(SPHERE_DETAIL)*sizeof(unsigned int));
  sphere_mesh->num_indices = SHAPES_SPHERE_INDICES(SPHERE_DETAIL);
  //sphere_mesh->num_indices -= 3;

  sphere_mesh->indices_type = GL_UNSIGNED_INT;
  //sphere_indices = (unsigned int*)malloc(SHAPES_SPHERE_INDICES(SPHERE_DETAIL)*sizeof(unsigned int));
  
  //shapes_sphere(sphere_vertices,NULL,NULL,sphere_indices,255.0,SPHERE_DETAIL);
  shapes_sphere(sphere_mesh->vertices,NULL,sphere_mesh->texels,(unsigned int*)sphere_mesh->indices,255.0,SPHERE_DETAIL);
  
  
  mesh_upload(sphere_mesh);    
  /*
  unsigned int *apa = (unsigned int*)sphere_mesh->indices;
  for (int a = 0; a < sphere_mesh->num_indices; a ++) {
    printf(" %f, %f, %f \n",
	   sphere_mesh->vertices[apa[a]].x,
	   sphere_mesh->vertices[apa[a]].y,   
	   sphere_mesh->vertices[apa[a]].z);
  }
  */
   
  /* 
  Image *t = image_create(2048,2048,3); 
  image_clear(t);
  unsigned char pcol[3] = {255,255,255};
  
  int x1, y1;
  int x2, y2; 
  int x3, y3;   
  int p = 0; 

  unsigned int *i = (unsigned int*)sphere_mesh->indices; 
  for (int points = 0; points < sphere_mesh->num_indices/3; points ++) {
    x1 = sphere_mesh->texcoords[i[p]].x * 2048.0;
    y1 = sphere_mesh->texcoords[i[p++]].y * 2048.0;

    x2 = sphere_mesh->texcoords[i[p]].x * 2048.0;
    y2 = sphere_mesh->texcoords[i[p++]].y * 2048.0;

    x3 = sphere_mesh->texcoords[i[p]].x * 2048.0;
    y3 = sphere_mesh->texcoords[i[p++]].y * 2048.0;
    
    //printf("%d, %d\n",xp,yp); 
    //image_setPixel(t,xp,yp,pcol);
    image_drawLine(t,x1,y1,x2,y2,pcol);
    image_drawLine(t,x2,y2,x3,y3,pcol);
    image_drawLine(t,x3,y3,x1,y1,pcol);
    //image_setPixel(t,x1,y1,pcol);
    //image_setPixel(t,x2,y2,pcol);
    //image_setPixel(t,x3,y3,pcol);
  }
  image_storeRaw("sphere_map.raw",t);
  image_destroy(t); 
  */
}  
  
 

// -----------------------------------------------------------------------------
// setShaders
// -----------------------------------------------------------------------------
void setShaders() {
	
  char *vs,*fs;
       
  /* setup shader1 ---------------------------------------------------------------------- */


  shader1_vert = shader_load("Shaders/const_mv.vert",
				 GL_VERTEX_SHADER);
  shader1_frag = shader_load("Shaders/const_mv.frag",
				 GL_FRAGMENT_SHADER);


  glCompileShader(shader1_vert);
  shader_printCompileLog(shader1_vert); 
  glCompileShader(shader1_frag); 
  shader_printCompileLog(shader1_frag);

  shader1_program = glCreateProgram();
  //printf("%d \n", shader1_program);
		
  glAttachShader(shader1_program,shader1_vert);
  glAttachShader(shader1_program,shader1_frag);
  printGLError("setShader8:");

  glLinkProgram(shader1_program);
  
  shader1_color_uniloc = glGetUniformLocation(shader1_program,"color");
  
  printf("shader1_color_uniloc: %d\n",shader1_color_uniloc);
  
  shader1_proj = glGetUniformLocation(shader1_program, "proj");
  shader1_mod = glGetUniformLocation(shader1_program, "mod");

  printGLError("setShader9:");
  shader1_vindex = glGetAttribLocation(shader1_program, "Vertex");


  /* setup shader2 ---------------------------------------------------------------------- */

  
  shader2_vert = shader_load("Shaders/image2d_alpha_mv.vert",
				 GL_VERTEX_SHADER);
  shader2_frag = shader_load("Shaders/image2d_alpha_mv.frag",
				 GL_FRAGMENT_SHADER);


  glCompileShader(shader2_vert);
  shader_printCompileLog(shader2_vert);
  glCompileShader(shader2_frag);
  shader_printCompileLog(shader2_frag);
   
  shader2_program = glCreateProgram();

		
  glAttachShader(shader2_program,shader2_vert);
  glAttachShader(shader2_program,shader2_frag);
  printGLError("setShader8:");

  glLinkProgram(shader2_program);
  
 

  shader2_proj = glGetUniformLocation(shader2_program, "proj");
  shader2_mod = glGetUniformLocation(shader2_program, "mod");
  shader2_texSampler = glGetUniformLocation(shader2_program, "tex");

  printGLError("setShader9:");
  shader2_vindex = glGetAttribLocation(shader2_program, "Vertex");
  shader2_tindex = glGetAttribLocation(shader2_program, "TexCoord0");

 

 
  /* setup shader for sphere_mesh */
  sphere_unis[0].type = UNIFORM_MAT4X4F;
  sphere_unis[0].id   = shader1_proj;
  sphere_unis[0].data.m4x4f = &projectionMatrix;
 
  sphere_unis[1].type = UNIFORM_MAT4X4F;
  sphere_unis[1].id   = shader1_mod;
  sphere_unis[1].data.m4x4f = &modelViewMatrix;
 
  sphere_unis[2].type = UNIFORM_VEC4F;
  sphere_unis[2].id   = shader1_color_uniloc;
  sphere_unis[2].data.v4f = &mesh_color;

  /*
  sphere_mesh->uniforms = sphere_unis; 
  sphere_mesh->num_uniforms = 3;
  sphere_mesh->shader   = shader1_program;
  sphere_mesh->attributes[VERTEX_INDEX].vattrib = shader1_vindex;
  */
  shader1.uniforms = sphere_unis; 
  shader1.num_uniforms = 3;
  shader1.shader   = shader1_program;
  shader1.attributes[VERTEX_INDEX].vattrib = shader1_vindex;
  shader1.attributes[VERTEX_INDEX].active = true;


  /* setup shader for ground_mesh */
  
  ground_unis[0].type = UNIFORM_MAT4X4F;
  ground_unis[0].id   = shader2_proj;
  ground_unis[0].data.m4x4f = &projectionMatrix;
  
  ground_unis[1].type = UNIFORM_MAT4X4F;
  ground_unis[1].id   = shader2_mod;
  ground_unis[1].data.m4x4f = &modelViewMatrix;
 
  ground_unis[2].type = UNIFORM_INT;
  ground_unis[2].id   = shader2_texSampler;
  ground_unis[2].data.i = &texunit;
  /*  
  ground_mesh->uniforms = ground_unis; 
  ground_mesh->num_uniforms = 3;
  ground_mesh->shader   = shader2_program;
  ground_mesh->attributes[VERTEX_INDEX].vattrib = shader2_vindex;
  ground_mesh->attributes[TEXCOORD_INDEX].vattrib = shader2_tindex;
  */
  shader2.uniforms = ground_unis; 
  shader2.num_uniforms = 3;
  shader2.shader   = shader2_program;
  shader2.attributes[VERTEX_INDEX].vattrib = shader2_vindex;
  shader2.attributes[VERTEX_INDEX].active = true;
  shader2.attributes[TEXEL_INDEX].vattrib = shader2_tindex;
  shader2.attributes[TEXEL_INDEX].active = true;
  

      



 
}
// -----------------------------------------------------------------------------
// drawLine
// -----------------------------------------------------------------------------
void drawLine(Vector3f *from, Vector3f *to, Vector3f *color) {
  /* set appropriate shader */
  glUseProgram(shader1_program);

  printGLError("drawLine0:");

  glUniformMatrix4fv(shader1_proj, 1, GL_FALSE, projectionMatrix);
  printGLError("drawLine1:");
  glUniformMatrix4fv(shader1_mod, 1, GL_FALSE,  modelViewMatrix);
  printGLError("drawLine2:");
  glUniform4f(shader1_color_uniloc,color->x,color->y,color->z,1.0f);
  
  GLuint line; 
  Vector3f endpoints[2] = 
    {{from->x, from->y, from->z},
     {to->x, to->y, to->z}};
 
 
  glEnableVertexAttribArray(shader1_vindex);

  glGenBuffers(1, &line);
  glBindBuffer(GL_ARRAY_BUFFER, line);
  glBufferData(GL_ARRAY_BUFFER, 2*sizeof(Vector3f), endpoints, GL_DYNAMIC_DRAW);
 
  glVertexAttribPointer(shader1_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);

  //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pyr_fac);

  printGLError("drawLine3:");
 
 
  glDrawArrays(GL_LINES, 0, 2);
  
  

  //glDrawElements(GL_LINE_LOOP, 9, GL_UNSIGNED_BYTE, (void*)0);
  glDisableVertexAttribArray(shader1_vindex);

}


// -----------------------------------------------------------------------------
// drawMesh
// -----------------------------------------------------------------------------
void drawDotMesh() {
  /* set appropriate shader */
  glUseProgram(shader1_program);

  printGLError("drawDotMesh0:");

  glUniformMatrix4fv(shader1_proj, 1, GL_FALSE, projectionMatrix);
  printGLError("drawDotMesh1:");
  glUniformMatrix4fv(shader1_mod, 1, GL_FALSE,  modelViewMatrix);
  printGLError("drawDotMesh2:");
  glUniform4f(shader1_color_uniloc,1.0f,0.0f,0.0f,1.0f);

 

  glEnableVertexAttribArray(shader1_vindex);

  glBindBuffer(GL_ARRAY_BUFFER, mesh_vert);   
  glVertexAttribPointer(shader1_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);

  //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pyr_fac);

  printGLError("drawDotMesh3:");

 
  glDrawArrays(GL_POINTS, 0, 512*512);
  
  

  //glDrawElements(GL_LINE_LOOP, 9, GL_UNSIGNED_BYTE, (void*)0);
  glDisableVertexAttribArray(shader1_vindex);
}

void drawFilledMesh() {
  /* set appropriate shader */
  glUseProgram(shader1_program);

  printGLError("drawFilledMesh0:");

  glUniformMatrix4fv(shader1_proj, 1, GL_FALSE, projectionMatrix);
  printGLError("drawFilledMesh1:");
  glUniformMatrix4fv(shader1_mod, 1, GL_FALSE,  modelViewMatrix);
  printGLError("drawFilledMesh2:");
  glUniform4f(shader1_color_uniloc,1.0f,1.0f,1.0f,1.0f);

 

  glEnableVertexAttribArray(shader1_vindex);

  glBindBuffer(GL_ARRAY_BUFFER, mesh_vert);   
  glVertexAttribPointer(shader1_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);

  //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pyr_fac);

  printGLError("drawFilledMesh3:");


  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_ix);
  //glDrawElements(GL_TRIANGLE_STRIP, 6, GL_UNSIGNED_BYTE, 0);

  glDrawElements(GL_TRIANGLES,512*512*6,GL_UNSIGNED_INT,0);
  printGLError("drawFilledMesh4:");
  

  //glDrawElements(GL_LINE_LOOP, 9, GL_UNSIGNED_BYTE, (void*)0);
  glDisableVertexAttribArray(shader1_vindex);
}


void drawTexturedMesh() {
  /* set appropriate shader */
  glUseProgram(shader2_program);
  printGLError("drawTMesh-2:");
  glActiveTexture(GL_TEXTURE0);
  printGLError("drawTMesh-1:");
  glBindTexture(GL_TEXTURE_2D, texture1);
  glUniform1i(shader2_texSampler,0);


  printGLError("drawTMesh0:");

  glUniformMatrix4fv(shader2_proj, 1, GL_FALSE, projectionMatrix);
  printGLError("drawTMesh1:");
  glUniformMatrix4fv(shader2_mod, 1, GL_FALSE,  modelViewMatrix);
  printGLError("drawTMesh2:");
  

 

  glEnableVertexAttribArray(shader2_vindex);

  glBindBuffer(GL_ARRAY_BUFFER, mesh_vert);   
  glVertexAttribPointer(shader2_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);


  glEnableVertexAttribArray(shader2_tindex);

  glBindBuffer(GL_ARRAY_BUFFER, mesh_tex);   
  glVertexAttribPointer(shader2_tindex,2,GL_FLOAT,GL_FALSE,0,(void*)0);

  //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pyr_fac);

  printGLError("drawTMesh3:");


  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_ix);
  //glDrawElements(GL_TRIANGLE_STRIP, 6, GL_UNSIGNED_BYTE, 0);

  glDrawElements(GL_TRIANGLES,511*511*6,GL_UNSIGNED_INT,0);
  printGLError("drawTMesh4:");
  

  //glDrawElements(GL_LINE_LOOP, 9, GL_UNSIGNED_BYTE, (void*)0);
  glDisableVertexAttribArray(shader2_vindex);
  glDisableVertexAttribArray(shader2_tindex);
}
// -----------------------------------------------------------------------------
// drawDisc
// -----------------------------------------------------------------------------


void drawFilledDisc() {
  /* set appropriate shader */
  glUseProgram(shader1_program);

  printGLError("drawDisc0:");

  glUniformMatrix4fv(shader1_proj, 1, GL_FALSE, projectionMatrix);
  printGLError("drawDisc1:");
  glUniformMatrix4fv(shader1_mod, 1, GL_FALSE,  modelViewMatrix);
  printGLError("drawDisc2:");
  glUniform4f(shader1_color_uniloc,1.0f,1.0f,1.0f,1.0f);

 
  glEnableVertexAttribArray(shader1_vindex);

  glBindBuffer(GL_ARRAY_BUFFER, disc_vert);   
  glVertexAttribPointer(shader1_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);
  printGLError("drawDisc3:");


  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, disc_ix);
  glDrawElements(GL_TRIANGLES,300,GL_UNSIGNED_SHORT,0);
  printGLError("drawDisc4:");
  

 
  glDisableVertexAttribArray(shader1_vindex);
}
void drawTexturedDisc() {
  /* set appropriate shader */
  glUseProgram(shader2_program);

  glActiveTexture(GL_TEXTURE0);
 
  glBindTexture(GL_TEXTURE_2D, texture2);
  glUniform1i(shader2_texSampler,0);


  printGLError("drawTDisc0:");

  glUniformMatrix4fv(shader2_proj, 1, GL_FALSE, projectionMatrix);
  printGLError("drawTDisc1:");
  glUniformMatrix4fv(shader2_mod, 1, GL_FALSE,  modelViewMatrix);
  printGLError("drawTDisc2:");
  

 

  glEnableVertexAttribArray(shader2_vindex);

  glBindBuffer(GL_ARRAY_BUFFER, disc_vert);   
  glVertexAttribPointer(shader2_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);


  glEnableVertexAttribArray(shader2_tindex);

  glBindBuffer(GL_ARRAY_BUFFER, disc_tex);   
  glVertexAttribPointer(shader2_tindex,2,GL_FLOAT,GL_FALSE,0,(void*)0);

  //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pyr_fac);

  printGLError("drawTDisc3:");


  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, disc_ix);
  //glDrawElements(GL_TRIANGLE_STRIP, 6, GL_UNSIGNED_BYTE, 0);

  glDrawElements(GL_TRIANGLES,300,GL_UNSIGNED_SHORT,0);
  printGLError("drawTDisc4:");
  

  //glDrawElements(GL_LINE_LOOP, 9, GL_UNSIGNED_BYTE, (void*)0);
  glDisableVertexAttribArray(shader2_vindex);
  glDisableVertexAttribArray(shader2_tindex);
}
// -----------------------------------------------------------------------------
// drawCylinder
// -----------------------------------------------------------------------------
void drawDotCylinder() {
  /* set appropriate shader */
  glUseProgram(shader1_program);
 
  printGLError("DrawDotCylinder0:");

  glUniformMatrix4fv(shader1_proj, 1, GL_FALSE, projectionMatrix);
  printGLError("DrawDotCylinder1:");
  glUniformMatrix4fv(shader1_mod, 1, GL_FALSE,  modelViewMatrix);
  printGLError("DrawDotCylinder2:");
  glUniform4f(shader1_color_uniloc,1.0f,0.0f,0.0f,1.0f);

 

  glEnableVertexAttribArray(shader1_vindex);

  glBindBuffer(GL_ARRAY_BUFFER, cyl_vert);   
  glVertexAttribPointer(shader1_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);

  //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pyr_fac);

  printGLError("DrawDotCylinder3:");

 
  glDrawArrays(GL_POINTS, 0, 100);
  
  

  //glDrawElements(GL_LINE_LOOP, 9, GL_UNSIGNED_BYTE, (void*)0);
  glDisableVertexAttribArray(shader1_vindex);
}



// -----------------------------------------------------------------------------
// drawSphere
// -----------------------------------------------------------------------------
void drawDotSphere() {
  /* set appropriate shader */
  glUseProgram(shader1_program);

  printGLError("DrawDotSphere0:");

  glUniformMatrix4fv(shader1_proj, 1, GL_FALSE, projectionMatrix);
  printGLError("DrawDotSphere1:");
  glUniformMatrix4fv(shader1_mod, 1, GL_FALSE,  modelViewMatrix);
  printGLError("DrawDotSphere2:");
  glUniform4f(shader1_color_uniloc,1.0f,0.0f,0.0f,1.0f);

 

  glEnableVertexAttribArray(shader1_vindex);

  glBindBuffer(GL_ARRAY_BUFFER, sphere_vert);   
  glVertexAttribPointer(shader1_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);

  //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pyr_fac);

  printGLError("DrawDotSphere3:");

 
  glDrawArrays(GL_POINTS, 0, SHAPES_SPHERE_INDICES(SPHERE_DETAIL));
  
  

  //glDrawElements(GL_LINE_LOOP, 9, GL_UNSIGNED_BYTE, (void*)0);
  glDisableVertexAttribArray(shader1_vindex);
}
void drawDotSpherei(int i) {
  /* set appropriate shader */
  glUseProgram(shader1_program);

  printGLError("DrawDotSphere0:");

  glUniformMatrix4fv(shader1_proj, 1, GL_FALSE, projectionMatrix);
  printGLError("DrawDotSphere1:");
  glUniformMatrix4fv(shader1_mod, 1, GL_FALSE,  modelViewMatrix);
  printGLError("DrawDotSphere2:");
  glUniform4f(shader1_color_uniloc,1.0f,0.0f,0.0f,1.0f);

 

  glEnableVertexAttribArray(shader1_vindex);

  glBindBuffer(GL_ARRAY_BUFFER, sphere_vert);   
  glVertexAttribPointer(shader1_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);

  //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pyr_fac);

  printGLError("DrawDotSphere3:");

  
  glDrawArrays(GL_POINTS, 0, i);
  
   
 
  //glDrawElements(GL_LINE_LOOP, 9, GL_UNSIGNED_BYTE, (void*)0);
  glDisableVertexAttribArray(shader1_vindex);
}
 
void drawFilledSphere() {    
  /* set appropriate shader */     
  glUseProgram(shader1_program);


  glUniformMatrix4fv(shader1_proj, 1, GL_FALSE, projectionMatrix);

  glUniformMatrix4fv(shader1_mod, 1, GL_FALSE,  modelViewMatrix);

  glUniform4f(shader1_color_uniloc,1.0f,1.0f,1.0f,1.0f);

  
  glEnableVertexAttribArray(shader1_vindex);

  glBindBuffer(GL_ARRAY_BUFFER, sphere_vert);   
  glVertexAttribPointer(shader1_vindex,3,GL_FLOAT,GL_FALSE,0,(void*)0);

 
 
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ix);
  glDrawElements(GL_TRIANGLES,SHAPES_SPHERE_INDICES(SPHERE_DETAIL),GL_UNSIGNED_INT,0);
  
  
 
  glDisableVertexAttribArray(shader1_vindex);
}  
    
    
// -----------------------------------------------------------------------------
// display callback function
// -----------------------------------------------------------------------------
void display(void) {    
  glutSetWindow(win);  
  
  //  glClearColor(1.0,1.0,1.0,1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  matrix_identity4x4f(modelViewMatrix);

  
  Vector3f pos = vector3f(px,py,pz);
  Vector3f la  = vector3f(px+sin(3.14-r),py,pz+cos(3.14-r));
  Vector3f up  = vector3f(0.0,1.0,0.0);
  matrix_lookAtf(modelViewMatrix,&pos, &la  ,&up);

 
  //  printf("pos: %f, %f, %f\n", px,py,pz);
  //  printf("la : %f, %f, %f\n", la.x, la.y,la.z);
  //  printf("r  : %f \n",r);
 
  //drawTexturedMesh(); 
  
  mesh_renderTextured(&shader2,ground_mesh); 
  //mesh_renderFill(ground_mesh); 
  
  matrix_rotate4x4f(modelViewMatrix,2.14/2.0, 1.0,0.0,0.0);
  matrix_translate4x4f(modelViewMatrix,0.0,70.0,0.0);
  //drawFilledDisc() ;
  //drawTexturedDisc();
 
  //drawDotCylinder();
  //drawDotSphere(); 
  //drawDotSpherei(i); 
  //drawFilledSphere(); 
   
  //mesh_renderTextured(&shader2,sphere_mesh);
   
  printGLError("display:");  
  glutSwapBuffers();  
  printGLError("display:"); 
 
}     
void display2(void) { 
  //glutSetWindow(win2);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  matrix_identity4x4f(modelViewMatrix);
 
   
  Vector3f pos_real = {px,py,pz};
  Vector3f la_real  = {px+sin(3.14-r),py,pz+cos(3.14-r)};
  Vector3f up_real  = {0.0,1.0,0.0};
  //matrix_lookAtf(modelViewMatrix,&pos, &la  ,&up);
   
 
  Vector3f pos = {-255,100,455};
  Vector3f la  = {0,100,0};
  Vector3f up  = {0.0,1.0,0.0};
  matrix_lookAtf(modelViewMatrix,&pos, &la  ,&up);
 
  //  printf("pos: %f, %f, %f\n", px,py,pz);
  //  printf("la : %f, %f, %f\n", la.x, la.y,la.z);
  //  printf("r  : %f \n",r);
 
  //drawTexturedMesh(); 
  drawDotMesh(); 
  drawDotSphere(); 
  //matrix_rotate4x4f(modelViewMatrix,3.14/2.0, 1.0,0.0,0.0);
  matrix_translate4x4f(modelViewMatrix,px,py,pz);
  //drawFilledDisc();

  
 
  Vector3f a = vector3f(0.0,0.0,0.0);
  Vector3f b = vector3f(-px,-py,-pz);
  Vector3f c = vector3f(1.0,1.0,0.0);
  
  Vector3f ld; 
  ld.x = la_real.x - pos_real.x;
  ld.y = la_real.y - pos_real.y;
  ld.z = la_real.z - pos_real.z;
 
  vector_normalize3f(&ld,&ld);
  
  ld.x = ld.x * 5;
  ld.y = ld.y * 5;
  ld.z = ld.z * 5;
  
  Vector3f lu;
  lu.x = up_real.x * 5;
  lu.y = up_real.y * 5;
  lu.z = up_real.z * 5;

  drawLine(&ld,&a,&c);
  drawLine(&lu,&a,&c);
  
  

  //drawLine(&a,&b,&c);
	   
	   
  //drawTexturedDisc();

 
  printGLError("display:");
  glutSwapBuffers(); 
  printGLError("display:");

}

// -----------------------------------------------------------------------------
// reshape
// -----------------------------------------------------------------------------

void reshape(int w, int h) {

  window_x = w;
  window_y = h;

  
  float near = 2.0f;
  float far  = 1000.0f;
  float fov  = 45.0*3.141592/360.0; 
  float top   = tan(fov) * near;
  float bottom = -top;
  
  float aspect = 1.25;
  float left = aspect * bottom;
  float right = aspect * top;
 
  
  matrix_frustumf(projectionMatrix, left, right,bottom,top,near,far);
  
  glViewport(0,0,w,h); 
}
 


// -----------------------------------------------------------------------------
// init (set various things up)
// -----------------------------------------------------------------------------
void init(void)
{
   
  Image *img; 
  
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK); 
  glEnable(GL_DEPTH_TEST);


  sphere_mesh = mesh_create();
  ground_mesh = mesh_create();

  setShaders(); 
  initBuffer();
  
  /*  
  img = image_loadPNG("Pictures/texmap3.png");
  image_hFlip(img); 
  
  initTexture2D(texture1,1024,1024,GL_RGB,img->data);
  image_destroy(img);
  */

  img = image_loadPNG("Pictures/texmap3.png");
  image_hFlip(img); 
  
  initTexture2D(texture1,512,512,GL_RGB,img->data);
  image_destroy(img);



  img = image_loadPNG("Pictures/texmap2.png"); 
  image_hFlip(img);
  
  initTexture2D(texture2,512,512,GL_RGB,img->data);
  image_destroy(img);


  matrix_identity4x4f(modelViewMatrix);
  matrix_identity4x4f(projectionMatrix);
}

void timer(int value) {
  
  if (buttons[LEFT]) {
    r -= 0.015;
  }
  if (buttons[RIGHT]) {
    r += 0.015;        
  }
  if (buttons[DOWN]) {
    px -= 3*sin(3.14-r);
    pz -= 3*cos(3.14-r);
  }
  if (buttons[UP]) {
    px += 3*sin(3.14-r);
    pz += 3*cos(3.14-r);
  }

  // glutSetWindow(win);
  glutPostRedisplay(); 
  // glutSetWindow(win2);
  // glutPostRedisplay();
  glutTimerFunc(10,timer,0);
}


void timer2(int value) {
  i ++;
  if (i > 5000) i = 0;
  glutTimerFunc(100,timer2,0);
}

// -----------------------------------------------------------------------------
// Arrow keys
// -----------------------------------------------------------------------------
void ArrowKeyDown(int key, int x, int y){
  
  if (key == GLUT_KEY_LEFT) {
    //r -= 0.015;
    buttons[LEFT] = 1;
  }
  if (key == GLUT_KEY_RIGHT) {
    //r += 0.015;
    buttons[RIGHT] = 1;
  }
  if (key == GLUT_KEY_UP) {
    //px -= 3*sin(3.14-r);
    //pz -= 3*cos(3.14-r);
    buttons[UP] = 1;
  }
  if (key == GLUT_KEY_DOWN) {
    //px += 3*sin(3.14-r); 
    //pz += 3*cos(3.14-r);
    buttons[DOWN] = 1;
  } 
}
void ArrowKeyUp(int key, int x, int y){

  if (key == GLUT_KEY_LEFT)
    buttons[LEFT] = 0;
  if (key == GLUT_KEY_RIGHT) 
    buttons[RIGHT] = 0;
  if (key == GLUT_KEY_UP)
    buttons[UP] = 0;
  if (key == GLUT_KEY_DOWN)
    buttons[DOWN] = 0;   
}

void mouse(int button, int state, int x, int y) {
  
  int wx;
  int wy; 
  static int active = 1;
  if (state == GLUT_DOWN) {
    wx =  x; 
    wy =  y;  
    if (button == 0) py --;
    if (button == 2) py ++; 
    fprintf(stdout,"PRESSED %d %d\n",button, wy);
    glutPostRedisplay();
  }
  
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
 
  int v = 0;
  int m = 0; 
   

  //Initialize glut
  glutInit(&argc,argv);
  
  //Set graphics 
  glutInitDisplayMode(GLUT_RGB |  GLUT_DOUBLE | GLUT_DEPTH);
  
  // OpenGL 3.2
  glutInitContextVersion(3,2);
  glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
  
  
  //Create window 
  glutInitWindowSize(window_x,window_y); 
  win = glutCreateWindow("Gfx");
  glutPositionWindow(0,20);
  GLenum GlewInitResult;

  glewExperimental = GL_TRUE;
  GlewInitResult = glewInit();
  
  if (GLEW_OK != GlewInitResult) {
    fprintf(
	    stderr,
	    "ERROR: %s\n",
	    glewGetErrorString(GlewInitResult)
	    );
    exit(EXIT_FAILURE);
  }
  
  fprintf(
	  stdout,
	  "INFO: OpenGL Version: %s\n",
	  glGetString(GL_VERSION)
	  );
  
  ignoreErrors();
  
  init();

  //Register callbacks
  glutMouseFunc(mouse);
  glutReshapeFunc(reshape); 
  glutDisplayFunc(display);

  glutSpecialFunc(ArrowKeyDown);
  glutSpecialUpFunc(ArrowKeyUp);
  glutTimerFunc(10,timer,0);
  glutTimerFunc(100,timer2,0);
  
  
  printf("OpenGL version: %s\n",glGetString(GL_VERSION));
  
  glutMainLoop();
 
  
}
