// C99 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glew.h> 
#include <GL/freeglut.h> 
#include <GL/gl.h>

#include <math.h>

#include "Graphics/matrix.h"
#include "Graphics/frame.h"
#include "Graphics/image.h"
#include "Graphics/error.h"
#include "Graphics/shapes.h" 
#include "Graphics/texture.h"
#include "Graphics/mesh.h"
#include "Graphics/shader.h"

#define LEFT 0
#define RIGHT 1
#define UP 2 
#define DOWN 3
int buttons[4]; 

float r = 0.0;
float k = 0.0;

float px = 0.0;  
float pz = 10.0; 
float py = 13.0;

Frame cameraFrame; 

const float pi  = 3.141592653589793;
const float pi2 = 6.283185307179586;
// -----------------------------------------------------------------------------
// Globals.. 
// -----------------------------------------------------------------------------
int win;

/* Matrices, Projection, modelview and normal */
Mat4x4f projectionMatrix; 
Mat4x4f modelViewMatrix; 
Mat3x3f normalMatrix;  


GLuint texture1; 
GLuint texture2; 
GLint texunit = 0; //there is a difference between texture unit and texture!

int window_x = 800; 
int window_y = 600;

float rotation = 0.0;
Vector4f lightsource;

// -----------------------------------------------------------------------------
// Mesh and Shaders
// -----------------------------------------------------------------------------

Mesh *cube_mesh;

Uniform tmap_shader_unis[3];
Shader tmap_shader;
    
Mesh *wall_mesh; 

// --------------------------------------------------------------------------
// World 
// --------------------------------------------------------------------------

unsigned char world[64] = 
  {1,1,1,1,1,1,1,1,
   1,0,0,0,0,0,0,1,
   1,0,0,0,0,0,0,1,
   1,0,0,0,0,0,0,1,
   1,0,0,0,0,0,0,1,
   1,0,0,0,0,0,0,1,
   1,0,0,0,0,0,0,1,
   1,1,1,1,1,1,1,1};
   
void renderWall(float px, float pz) {
  Mat4x4f tmp; 
  
  memcpy(tmp,modelViewMatrix,sizeof(Mat4x4f));
  
  matrix_translate4x4f(modelViewMatrix,px,10,pz);
  mesh_renderTex_prim(&tmap_shader,texture2,wall_mesh);

  matrix_rotate4x4f(modelViewMatrix,pi/2.0,0,1,0);
  matrix_translate4x4f(modelViewMatrix,-10,0,10);
  mesh_renderTex_prim(&tmap_shader,texture2,wall_mesh);
  
  matrix_rotate4x4f(modelViewMatrix,pi,0,1,0);
  matrix_translate4x4f(modelViewMatrix,0,0,20);
  mesh_renderTex_prim(&tmap_shader,texture2,wall_mesh);

  matrix_rotate4x4f(modelViewMatrix,-pi/2,0,1,0);
  matrix_translate4x4f(modelViewMatrix,10,0,10);
  mesh_renderTex_prim(&tmap_shader,texture2,wall_mesh);

 
  memcpy(modelViewMatrix,tmp,sizeof(Mat4x4f));
   
  

} 

void renderWorld(int w, int h, unsigned char *world){

  Mat4x4f tmp; 
  

   
  for (int i = 0; i < w; ++i) {
    for (int j = 0; j < h; ++j) { 
      memcpy(tmp,modelViewMatrix,sizeof(Mat4x4f));
      matrix_translate4x4f(modelViewMatrix,i*20,0,j*20);
      if (world[i+j*8]) { 
	renderWall(0,0);
      }
      memcpy(modelViewMatrix,tmp,sizeof(Mat4x4f));
      
    }
  }

}


// --------------------------------------------------------------------------
// Initialize Meshes 
// --------------------------------------------------------------------------

void initMeshes() { 
  cube_mesh = mesh_create();
  
  mesh_allocateAttribs(cube_mesh,
		       400, 
		       MESH_VERTICES | MESH_TEXELS);
  
  cube_mesh->num_indices =  19*19*6; 
  cube_mesh->indices_type = GL_UNSIGNED_INT;
  cube_mesh->indices = malloc(19*19*6*sizeof(int));
    
  shapes_flatMesh(cube_mesh->vertices,
		  NULL,
		  cube_mesh->texels, 
		  (unsigned int*)cube_mesh->indices,
		  20,20,   // dimensions
		  20.0f,20.0f, // step size
		  1); 

  //printGLError("before upload");
  mesh_upload_prim(cube_mesh); 
  //printGLError("upload");
  //mesh_freeAttribs(cube_mesh);
  //printGLError("freeattribs");


  wall_mesh = mesh_create();
  mesh_allocateAttribs(wall_mesh,
		       4, 
		       MESH_VERTICES | MESH_TEXELS);
  wall_mesh->num_indices = 6;
  wall_mesh->indices_type = GL_UNSIGNED_BYTE;
  wall_mesh->indices = malloc(6);
  
  
  shapes_quad(wall_mesh->vertices, 
	      NULL, 
	      wall_mesh->texels, 
	      wall_mesh->indices,
	      10,10);
  
  mesh_upload_prim(wall_mesh);
  mesh_freeAttribs(wall_mesh);
  //printGLError("before leaving initmeshes");  
}

// --------------------------------------------------------------------------
// Shaders
// --------------------------------------------------------------------------

void initShaders() { 
  GLuint p,f,v;
  GLuint shader_vindex;
  GLuint shader_tindex;
  GLuint shader_proj;
  GLuint shader_mod;
  GLuint shader_texSampler;
  
  printGLError("initShaders0");
  
  v = shader_load("Shaders/tex_simple.vert",GL_VERTEX_SHADER);
  f = shader_load("Shaders/tex_simple.frag",GL_FRAGMENT_SHADER);

  printGLError("initShaders1");

  glCompileShader(v);
  glCompileShader(f);
  shader_printCompileLog(v);
  shader_printCompileLog(f);

  printGLError("initShaders2");
  p = glCreateProgram();
  glAttachShader(p,v);
  printGLError("initShaders3a");
  glAttachShader(p,f);
  printGLError("initShaders3b");
  
  glLinkProgram(p);
  printGLError("initShaders3c");

  glBindAttribLocation(p,0,"TexCoord0");
  glBindAttribLocation(p,1,"Vertex");
 
  glLinkProgram(p);
  printGLError("initShaders3c");
 
  glBindAttribLocation(p,1,"TexCoord0");
  glBindAttribLocation(p,0,"Vertex");
 
  glLinkProgram(p);
  printGLError("initShaders3c");
 
  glUseProgram(p);
  printGLError("initShaders3d");
  

  shader_proj = glGetUniformLocation(p, "proj");
  printGLError("initShaders4");
  shader_mod  = glGetUniformLocation(p, "mod");
  printGLError("initShaders5");
  shader_texSampler = glGetUniformLocation(p,"tex"); 
  printGLError("initShaders6");
  shader_vindex = glGetAttribLocation(p, "Vertex");  
  printGLError("initShaders7");
  shader_tindex = glGetAttribLocation(p, "TexCoord0");  
  printGLError("initShaders8");
  tmap_shader_unis[0].type = UNIFORM_MAT4X4F;
  tmap_shader_unis[0].id   = shader_proj;
  tmap_shader_unis[0].data.m4x4f = &projectionMatrix;
  
  tmap_shader_unis[1].type = UNIFORM_MAT4X4F;
  tmap_shader_unis[1].id   = shader_mod;
  tmap_shader_unis[1].data.m4x4f = &modelViewMatrix;

  tmap_shader_unis[2].type = UNIFORM_INT;
  tmap_shader_unis[2].id   = shader_texSampler;
  tmap_shader_unis[2].data.i = &texunit;
 

  
  tmap_shader.uniforms = tmap_shader_unis;
  tmap_shader.num_uniforms = 3;
  tmap_shader.shader = p;
  tmap_shader.attributes[VERTEX_INDEX].vattrib = shader_vindex;
  tmap_shader.attributes[VERTEX_INDEX].active  = true;
  tmap_shader.attributes[TEXEL_INDEX].vattrib = shader_tindex;
  tmap_shader.attributes[TEXEL_INDEX].active = true;
  tmap_shader.attributes[NORMAL_INDEX].active = false;
  tmap_shader.attributes[COLOR_INDEX].active = false;
  printf("%d %d\n", shader_vindex, shader_tindex);
  printf("%d %d\n", VERTEX_INDEX, TEXEL_INDEX);

  
   /* Experiment */
  char info[256];
  int l;
  int s;
  GLenum t;

  int num_attribs = 0;
  glGetProgramiv(p,GL_ACTIVE_ATTRIBUTES,&num_attribs);
  printf("active attribs: %d\n",num_attribs); 

  for (int i = 0; i < num_attribs; i ++) { 
    glGetActiveAttrib(p,i,256,&l,&s,&t,info);
    printf("attrib: %s %d %d %d\n",info,l,s,t);
  }
  /* Experiment ends */

  
}


// -----------------------------------------------------------------------------
// display callback function
// -----------------------------------------------------------------------------
void display(void) {    
 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  matrix_identity4x4f(modelViewMatrix);

  /* SET UP VIEW */
  Vector3f pos = vector3f(px,py,pz);
  Vector3f la  = vector3f(sin(3.14-r),0,cos(3.14-r));
  Vector3f up  = vector3f(0.0,1.0,0.0);
  frame_create(&cameraFrame,&pos, &up, &la);

  frame_cameraTransform(modelViewMatrix,&cameraFrame);
 
  /* RENDER STUFF */
 
  
  //renderWall(25,25);
  
  printGLError("before render");
  
  //floor?
  mesh_renderTex_prim(&tmap_shader,texture1,cube_mesh);
 
  matrix_translate4x4f(modelViewMatrix,0,0,-200);
  renderWorld(8,8,world);


  
  
  

  
  /* SWAP BUFFERS */
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
  
  float aspect = (float)w/(float)h;
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
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK); 
  glEnable(GL_DEPTH_TEST);
  
  Image *img;
  
  img = image_loadPNG("Pictures/floor_dark.png");
  image_hFlip(img);
  
  printGLError("BEFORE");
  glGenTextures(1,&texture1);
  glGenTextures(1,&texture2);
  printGLError("Texturing");
  
  

  
  printf("Texture: %d\n",texture1); 
  //initTexture2D(texture1, 512,512, GL_RGBA, img->data);
  texture_init2DGenMip(texture1,512,512,GL_RGBA, img->data);
  glTexParameteri(GL_TEXTURE_2D,
		  GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  
  
  image_destroy(img);
  
  img = image_loadPNG("Pictures/wall2.png");
  image_hFlip(img); 
  //initTexture2D(texture2, 512,512, GL_RGBA, img->data);
  texture_init2DGenMip(texture2, 512,512, GL_RGBA, img->data);
 
  glTexParameteri(GL_TEXTURE_2D,
		  GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  
  image_destroy(img);
  

  
  /* Meshes */ 
  
  
  initMeshes();
   printGLError("after initMeshes");
  initShaders();
   printGLError("after initShaders");


  
  
  /* matrices */ 
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
    px -= sin(3.14-r);
    pz -= cos(3.14-r);
  }
  if (buttons[UP]) {
    px += sin(3.14-r);
    pz += cos(3.14-r);
  }
 
  glutPostRedisplay(); 
  
  glutTimerFunc(10,timer,0);
}



// -----------------------------------------------------------------------------
// Arrow keys
// -----------------------------------------------------------------------------
void ArrowKeyDown(int key, int x, int y){
  
  if (key == GLUT_KEY_LEFT) 
    buttons[LEFT] = 1;
  if (key == GLUT_KEY_RIGHT) 
    buttons[RIGHT] = 1;
  if (key == GLUT_KEY_UP) 
    buttons[UP] = 1;
  if (key == GLUT_KEY_DOWN) 
    buttons[DOWN] = 1;
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
 
  //Initialize glut
  glutInit(&argc,argv);
  
  //Set graphics 
  glutInitDisplayMode(GLUT_RGB |  GLUT_DOUBLE | GLUT_DEPTH);
  printGLError("InitDispMode:");

  // OpenGL
  glutInitContextVersion(3,2);
  //glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
  //glutInitContextProfile(GLUT_CORE_PROFILE);
  printGLError("InitVersion:");

  glutSetOption(
	        GLUT_ACTION_ON_WINDOW_CLOSE,
	        GLUT_ACTION_GLUTMAINLOOP_RETURNS
		);

	   
  //Create window 
  glutInitWindowSize(window_x,window_y); 
  win = glutCreateWindow("Gfx");
  glutPositionWindow(0,20);
  
  printGLError("PositionWindow:");

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
  ignoreErrors();


  init();
 

  //Register callbacks
  glutMouseFunc(mouse);
  glutReshapeFunc(reshape); 
  glutDisplayFunc(display);

  glutSpecialFunc(ArrowKeyDown);
  glutSpecialUpFunc(ArrowKeyUp);
  glutTimerFunc(10,timer,0);

 
  
  // print some info
  printf("OpenGL version: %s\n",glGetString(GL_VERSION));
  
  glutMainLoop();
 
  
}
