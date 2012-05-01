// C99 


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/time.h>

#include <GL/glew.h>
#include <GL/freeglut.h> 

#include <math.h>

#include "Graphics/matrix.h"
#include "Graphics/image.h"
#include "Graphics/error.h"
#include "Graphics/shapes.h" 
#include "Graphics/texture.h"
#include "Graphics/mesh.h"
#include "Graphics/multimesh.h"
#include "Graphics/shader.h"

#include "Graphics/text.h"
#include "Graphics/tools.h"



#define LEFT 0
#define RIGHT 1
#define UP 2 
#define DOWN 3
int buttons[4]; 

float r = 0.0;
float k = 0.0;

float px = 0.0;  
float pz = 10.0; 
float py = 0.0;

float ofs = 1.0;

// -----------------------------------------------------------------------------
// Globals.. 
// -----------------------------------------------------------------------------
int win;
float fps = 0.0f;
unsigned int frames;
struct timeval t;


/* Matrices, Projection, modelview and normal */
Mat4x4f projectionMatrix; 
Mat4x4f modelViewMatrix; 
Mat3x3f normalMatrix;  

Mat4x4f shadowMatrix;

//GLint texture1  = 89;
//GLint texture2  = 90;
//GLint texture3  = 91;

GLuint  tt;

// SHADOW MAP STUFF ------------------------------------------------------------
#define SHADOW_MAP_SIZE 2048
GLuint shadowMapTexture;
GLuint fbo;


int window_x = 800; 
int window_y = 600;

float rotation = 0.0;

Vector4f lightsource1 = {10.0,15.0, 20.0,1.0};
Vector4f lightsource;
//Vector3f lightDir = {0.0,0.0,-1.0};

Shader *basic_shader;
// -----------------------------------------------------------------------------
// MESHes
// -----------------------------------------------------------------------------

Multimesh mausoleum;
Uniform  shader1_unis[7];
Shader   shader1;
Vector4f color = {1.0,1.0,1.0,1.0};

Mesh quad;
Uniform shader2_unis[3];
Shader  shader2;
GLuint  texunit = 0; 

CharSet *cs;

Image *textTexture; 

Shader shaderShadow; 
Uniform shaderShadow_unis[2];

// -----------------------------------------------------------------------------
// Create ShadowMap
// -----------------------------------------------------------------------------


void createShadowMap() {

  //setup shadowmap
  glGenTextures(1, &shadowMapTexture);
  glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
  glTexImage2D(GL_TEXTURE_2D, 
	       0, 
	       GL_DEPTH_COMPONENT, 
	       SHADOW_MAP_SIZE, 
	       SHADOW_MAP_SIZE, 
	       0, 
	       GL_DEPTH_COMPONENT, 
	       GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
	
  // Generate framebuffer object
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	
  // whats this ?
  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);
	
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowMapTexture, 0);
	
  // error checking ? 
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
      printf("Error creating fbo 0x%x", glCheckFramebufferStatus(GL_FRAMEBUFFER));
    }

  // set some properties of the newly created fbo
  glClearDepth(1.0f);
  //glDepthFunc(GL_LEQUAL);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);

  // Set standard framebuffer
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

}



// -----------------------------------------------------------------------------
// setShaders
// -----------------------------------------------------------------------------
void setShaders() {
	
  char *vs,*fs;
       
  
  GLuint shader_frag;
  GLuint shader_vert; 
  GLuint shader_program;
  


  GLuint shader_vindex;
  GLuint shader_nindex;
  GLuint shader_tindex;

  GLuint shader_proj;
  GLuint shader_mod;
  GLuint shader_normal_matrix;
  GLuint shader_lightpos;
  GLuint shader_texSampler;
  GLuint shader_shadowTexture;
  GLuint shader_shadowMatrix;

  GLuint shader_color;

  /* setup fog shader ---------------------------------------------------------------------- */

  
  //shader_vert = shader_load("Shaders/const_lit_fog_mv.vert",
  //				 GL_VERTEX_SHADER);
  //shader_frag = shader_load("Shaders/const_lit_fog_mv.frag",
  //				 GL_FRAGMENT_SHADER);

  shader_vert = shader_load("Shaders/const_lit_fog_shadow_mv.vert",
				 GL_VERTEX_SHADER);
  shader_frag = shader_load("Shaders/const_lit_fog_shadow_mv.frag",
				 GL_FRAGMENT_SHADER);


  glCompileShader(shader_vert);
  shader_printCompileLog(shader_vert);
  glCompileShader(shader_frag);
  shader_printCompileLog(shader_frag);
  shader_program = glCreateProgram();

		
  glAttachShader(shader_program,shader_vert); 
  glAttachShader(shader_program,shader_frag);


  glLinkProgram(shader_program); 
     
   

  shader_proj = glGetUniformLocation(shader_program, "proj");
  shader_mod = glGetUniformLocation(shader_program, "mod");
  shader_color = glGetUniformLocation(shader_program, "Color");
  //shader_texSampler = glGetUniformLocation(shader_program, "tex");
  shader_normal_matrix = glGetUniformLocation(shader_program,"normalMatrix");
  shader_lightpos = glGetUniformLocation(shader_program,"lightPos");
  shader_shadowTexture = glGetUniformLocation(shader_program,"shadowTexture");
  shader_shadowMatrix  = glGetUniformLocation(shader_program,"shadowMatrix");
 
    
  printGLError("setShader9:");
  shader_vindex = glGetAttribLocation(shader_program, "Vertex");
  //shader_tindex = glGetAttribLocation(shader_program, "TexCoord0");
  shader_nindex = glGetAttribLocation(shader_program, "Normal");

  shader_uniform_m4x4f(&shader1_unis[0],shader_proj,&projectionMatrix);
  shader_uniform_m4x4f(&shader1_unis[1],shader_mod,&modelViewMatrix);
  shader_uniform_v4f  (&shader1_unis[2],shader_color,&color);
  shader_uniform_v3f  (&shader1_unis[3],shader_lightpos,VECTOR_SUB3F(&lightsource));
  shader_uniform_m3x3f(&shader1_unis[4],shader_normal_matrix,&normalMatrix);
  shader_uniform_i    (&shader1_unis[5],shader_shadowTexture,&texunit);//&shadowMapTexture);
  shader_uniform_m4x4f(&shader1_unis[6],shader_shadowMatrix,&shadowMatrix);
  
  shader1.uniforms = shader1_unis;
  shader1.num_uniforms = 7; 
  
  shader1.shader   = shader_program;
  shader1.attributes[VERTEX_INDEX].vattrib = shader_vindex;
  shader1.attributes[VERTEX_INDEX].active = true;
  
  shader1.attributes[NORMAL_INDEX].vattrib = shader_nindex;
  shader1.attributes[NORMAL_INDEX].active = true;

  /* setup texturing shader ---------------------------------------------------------------------- */

  
  shader_vert = shader_load("Shaders/image2d_alpha_mv.vert",
				 GL_VERTEX_SHADER);
  shader_frag = shader_load("Shaders/image2d_alpha_mv.frag",
				 GL_FRAGMENT_SHADER);


  glCompileShader(shader_vert);
  shader_printCompileLog(shader_vert);
  glCompileShader(shader_frag);
  shader_printCompileLog(shader_frag);
   
  shader_program = glCreateProgram();

		
  glAttachShader(shader_program,shader_vert);
  glAttachShader(shader_program,shader_frag);

  glLinkProgram(shader_program); 
     
   

  shader_proj = glGetUniformLocation(shader_program, "proj");
  shader_mod = glGetUniformLocation(shader_program, "mod");
  //shader_color = glGetUniformLocation(shader_program, "Color");
  shader_texSampler = glGetUniformLocation(shader_program, "tex");
 
    

  shader_vindex = glGetAttribLocation(shader_program, "Vertex");
  shader_tindex = glGetAttribLocation(shader_program, "TexCoord0");
  //shader_nindex = glGetAttribLocation(shader_program, "Normal");

  shader_uniform_m4x4f(&shader2_unis[0],shader_proj,&projectionMatrix);
  shader_uniform_m4x4f(&shader2_unis[1],shader_mod,&modelViewMatrix);
  //shader_uniform_v4f  (&shader1_unis[2],shader_color,&color);
  shader_uniform_i(&shader2_unis[2],shader_texSampler,&texunit);

  shader2.uniforms = shader2_unis;
  shader2.num_uniforms = 3; 
  
  shader2.shader   = shader_program;
  shader2.attributes[VERTEX_INDEX].vattrib = shader_vindex;
  shader2.attributes[VERTEX_INDEX].active = true;
  
  shader2.attributes[TEXEL_INDEX].vattrib = shader_tindex;
  shader2.attributes[TEXEL_INDEX].active = true;
 
  

  /* setup shadowmapping shader ---------------------------------------------------------------------- */
  
  
  shader_vert = shader_load("Shaders/shadowmap.vert",
				 GL_VERTEX_SHADER);
  shader_frag = shader_load("Shaders/shadowmap.frag",
				 GL_FRAGMENT_SHADER);


  glCompileShader(shader_vert);
  shader_printCompileLog(shader_vert);
  glCompileShader(shader_frag);
  shader_printCompileLog(shader_frag);
   
  shader_program = glCreateProgram();

		
  glAttachShader(shader_program,shader_vert);
  glAttachShader(shader_program,shader_frag);

  glLinkProgram(shader_program); 
     
   

  shader_proj = glGetUniformLocation(shader_program, "proj");
  shader_mod = glGetUniformLocation(shader_program, "mod");

  shader_vindex = glGetAttribLocation(shader_program, "Vertex");

  shader_uniform_m4x4f(&shaderShadow_unis[0],shader_proj,&projectionMatrix);
  shader_uniform_m4x4f(&shaderShadow_unis[1],shader_mod,&modelViewMatrix);

  shaderShadow.uniforms = shaderShadow_unis;
  shaderShadow.num_uniforms = 2; 
  
  shaderShadow.shader   = shader_program;
  shaderShadow.attributes[VERTEX_INDEX].vattrib = shader_vindex;
  shaderShadow.attributes[VERTEX_INDEX].active = true;
  

    

 
} 

    
// -----------------------------------------------------------------------------
// display callback function
// -----------------------------------------------------------------------------
void display(void) {     
  
  Vector3f pos;
  Vector3f la; 
  Vector3f up; 
  
  Mat4x4f bias = {0.5f, 0.0f, 0.0f, 0.0f, 
		  0.0f, 0.5f, 0.0f, 0.0f,
		  0.0f, 0.0f, 0.5f, 0.0f,
		  0.5f, 0.5f, 0.5f, 1.0f};


  
  frames++;
  struct timeval now;
  gettimeofday(&now,NULL);
  unsigned long usecs = 
    (now.tv_sec * 1000000 + now.tv_usec) - 
    (t.tv_sec * 1000000 + t.tv_usec);


  if (usecs > 1000000) {
    fps = (float)frames;
    frames = 0;
    gettimeofday(&t,NULL);
  }
  
 
  
  matrix_identity4x4f(modelViewMatrix); 


  /* set projection */

  float near = 1.0f; 
  float far  = 1000.0f;
  float fov  = 90.0*3.141592/360.0; 
  //float fov  = 181.0f*3.141592/360.0; 
  float top   = tan(fov) * near;
  float bottom = -top;
  
  float aspect = 1.0f;
  float left = aspect * bottom;
  float right = aspect * top; 
 
  matrix_identity4x4f(projectionMatrix);
  matrix_frustumf(projectionMatrix, left, right,bottom,top,near,far);



  /* RENDER INTO SHADOWMAP */
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glViewport(0,0,SHADOW_MAP_SIZE,SHADOW_MAP_SIZE);
  glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  glClear(GL_DEPTH_BUFFER_BIT);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);

 
  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset( 1.1, 4.0);

  
  /* SETUP VIEW AT THE LIGHT SOURCE */
  pos = vector3f(lightsource1.x,lightsource1.y,lightsource1.z);
  la  = vector3f(0.0,1.0,0.0);
  up  = vector3f(0.0,1.0,0.0);
 
  matrix_lookAtf(modelViewMatrix,&pos, &la  ,&up); 
 
  multimesh_renderFill(&shaderShadow,&mausoleum);

  
  matrix_identity4x4f(shadowMatrix);

  matrix_mul4x4f(shadowMatrix,shadowMatrix,bias); 
  matrix_mul4x4f(shadowMatrix,shadowMatrix,projectionMatrix);
  matrix_mul4x4f(shadowMatrix,shadowMatrix,modelViewMatrix);
 
 
 
  
  /* ---------------------------------------------------------------
     ------------------------------------------------------------ */
  /*
  Image *test = image_create(1024,1024,2); 
  Image *grey; 
 
  
  glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, test->data);

  grey = image_doubleByteToGrey(test);
  image_storeRaw("apa.raw",grey);
  image_destroy(test);
  image_destroy(grey);
  */
  /* ---------------------------------------------------------------
     ------------------------------------------------------------ */

  
  /* Render into colorbuffer */
 
  fov  = 45.0*3.141592/360.0;  
  top   = tan(fov) * near;
  bottom = -top;
  
  aspect = (float)window_x/(float)window_y;
  left = aspect * bottom;
  right = aspect * top; 
 
  matrix_identity4x4f(projectionMatrix);
  matrix_frustumf(projectionMatrix, left, right,bottom,top,near,far);


  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0,0,window_x,window_y); 
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glDisable(GL_POLYGON_OFFSET_FILL);
  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


  /* SET UP VIEW AT THE CAMERA*/ 
  pos = vector3f(px,py,pz);
  la  = vector3f(px+sin(3.14-r),py,pz+cos(3.14-r)); 
  up  = vector3f(0.0,1.0,0.0);
 
  matrix_identity4x4f(modelViewMatrix);
  matrix_lookAtf(modelViewMatrix,&pos, &la  ,&up);
     
 

  
  /*translate the lightsource */
  matrix_transform4x4f(&lightsource,modelViewMatrix,&lightsource1);

  //printf("%f %f %f\n", lightsource.x, lightsource.y, lightsource.z);
  
 
  //matrix_translate4x4f(modelViewMatrix,0.0f,0.0f,-10.0f);
  //matrix_rotate4x4f(modelViewMatrix,-rotation, 0.0,1.0,0.0);

  /* transform the normals */
  matrix_sub3x3f(normalMatrix,modelViewMatrix);
  matrix_invert3x3f(normalMatrix,normalMatrix);
  matrix_transpose3x3f(normalMatrix,normalMatrix);
  
  //mesh_renderFillLit(&monkey_shader,monkey_mesh);
  //matrix_scale4x4f(modelViewMatrix,20.0,20.0,20.0);

  //Mat4x4f modInv; 
  //matrix_invert4x4f(modInv,modelViewMatrix);
  //matrix_mul4x4f(shadowMatrix,shadowMatrix,modInv);
 
  
  //multimesh_renderFillLit(&shader1,&mausoleum);
  
  for (int i = 0; i < 9; i ++) {
    mausoleum.meshes[i].texture_id = shadowMapTexture;
    mausoleum.meshes[i].textured = true;
    
  }
  mesh_renderTextured(&shader1,&mausoleum.meshes[0]);
  mesh_renderTextured(&shader1,&mausoleum.meshes[1]);
  mesh_renderTextured(&shader1,&mausoleum.meshes[2]);  
  mesh_renderTextured(&shader1,&mausoleum.meshes[3]);
  mesh_renderTextured(&shader1,&mausoleum.meshes[4]);
  mesh_renderTextured(&shader1,&mausoleum.meshes[5]);
  mesh_renderTextured(&shader1,&mausoleum.meshes[6]);
  mesh_renderTextured(&shader1,&mausoleum.meshes[7]); 
  mesh_renderTextured(&shader1,&mausoleum.meshes[8]); 
  
  Vector3f v1 = {0.0,0.0,0.0 };
  Vector3f v2 = {100.0,100.0,100.0 };
  Vector3f c = {1.0,0.0,0.0};

  tools_drawLine(basic_shader,&v1,VECTOR_SUB3F(&lightsource1),&c);
  
  

 
  char fps_string[256]; 
  snprintf(fps_string,256,"FPS: %f",fps);
  image_clear(textTexture); 
  text_putStr(cs,fps_string,0,0,textTexture); 
  image_hFlip(textTexture);
  initTexture2D(tt,textTexture->width,textTexture->height,GL_RGBA,textTexture->data);


  
  
  matrix_ortho2Df(projectionMatrix,0,512,0,512);
  glViewport(0,0,window_x,window_y);  
   
  matrix_identity4x4f(modelViewMatrix);
  matrix_translate4x4f(modelViewMatrix,256,256,0.0);
  //matrix_scale4x4f(modelViewMatrix,0.25,0.25,0.25);
  quad.texture_id = tt;
  mesh_renderTexturedLit(&shader2,&quad);

  
 

  glutSwapBuffers();  
  printGLError("display:");    
 
}      
 
// -----------------------------------------------------------------------------
// reshape
// -----------------------------------------------------------------------------

void reshape(int w, int h) {

  window_x = w; 
  window_y = h;
   
  /*  
  float near = 1.0f; 
  float far  = 1000.0f;
  float fov  = 45.0*3.141592/360.0; 
  //float fov  = 181.0f*3.141592/360.0; 
  float top   = tan(fov) * near;
  float bottom = -top;
  
  float aspect = (float)w/(float)h;
  float left = aspect * bottom;
  float right = aspect * top; 
 
  
  matrix_frustumf(projectionMatrix, left, right,bottom,top,near,far);
  */

} 
 

 
// -----------------------------------------------------------------------------
// init (set various things up)
// -----------------------------------------------------------------------------
void init(void)
{
   
  Image *img;  
   
  glEnable(GL_CULL_FACE);
  //glCullFace(GL_BACK);  
  glEnable(GL_DEPTH_TEST); 
  glClearColor(0.5,0.5,0.5,1.0);

  multimesh_load3ds(&mausoleum, "mausoleum.3ds", MESH_LOAD_NORMALS);// | MESH_LOAD_TEXELS);  
  
  multimesh_upload(&mausoleum);    
    
  
  /* 
      Create quad mesh
   */ 
  mesh_init(&quad);
  
  Vector3f quadvertices[4];
  Vector2f quadtexels[4];
  unsigned char quadindices[6];

  glGenTextures(1, &tt);
  
  shapes_quad(quadvertices,NULL,quadtexels,quadindices,256,256);
  quad.vertices = quadvertices;
  quad.texels   = quadtexels;
  quad.num_vertices = 4;
  quad.indices = quadindices;
  quad.num_indices = 6;
  quad.indices_type = GL_UNSIGNED_BYTE;
  quad.textured = true;
  quad.texture_id = tt;
  
  mesh_upload(&quad);
  
  /* 
     
   */ 


  createShadowMap();


  setShaders(); 
  basic_shader = shader_basic(&modelViewMatrix,&projectionMatrix);

  //create shadowmap
  //createShadowMap();

 
  /*
  img = image_loadPNG("Pictures/earth1.png");
  image_hFlip(img); 
  
  initTexture2D(texture1,img->width,img->height,GL_RGB,img->data);
  image_destroy(img);
  
  img = image_loadPNG("Pictures/sun1.png"); 
  image_hFlip(img);  
    
  initTexture2D(texture2,img->width,img->height,GL_RGB,img->data);
  image_destroy(img);
  */
 
  cs = text_loadCharSet("Pictures/ASCIIWhite.raw",32,32);
  
  
  textTexture = image_create(1024,1024,4);
  image_clear(textTexture);



  /*
    
  char abc2[] = "abcdefghijklmnop"; 
  text_putStr(cs,abc1,0,0,textTexture); 
  text_putStr(cs,abc2,0,1,textTexture); 
  text_putStr(cs,abc1,0,2,textTexture); 
  text_putStr(cs,abc2,0,3,textTexture);  
  text_putStr(cs,abc1,0,4,textTexture); 
  text_putStr(cs,abc2,0,5,textTexture); 
  text_putStr(cs,abc1,0,6,textTexture); 
  text_putStr(cs,abc2,0,7,textTexture); 


  image_storeRaw("apa.raw",textTexture);
  */
  

  matrix_identity4x4f(modelViewMatrix);
  matrix_identity4x4f(projectionMatrix);

  
  gettimeofday(&t,NULL);
  
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

  rotation += 0.005;
  //ofs += 0.5;
  glutTimerFunc(10,timer,0);
}



// -----------------------------------------------------------------------------
// Arrow keys
// -----------------------------------------------------------------------------
void ArrowKeyDown(int key, int x, int y){
  
  if (key == GLUT_KEY_LEFT) {
    buttons[LEFT] = 1;
  }
  if (key == GLUT_KEY_RIGHT) {
    buttons[RIGHT] = 1;
  }
  if (key == GLUT_KEY_UP) {
    buttons[UP] = 1;
  }
  if (key == GLUT_KEY_DOWN) {
    buttons[DOWN] = 1;
  }
  if (key == GLUT_KEY_F1) {
    ofs += 0.01;
    printf("OFFSET: %f\n", ofs);
  }
  if (key == GLUT_KEY_F2) {
    ofs -= 0.01;
    printf("OFFSET: %f\n", ofs);
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
 
  }
  
}

void idle(){

  glutPostRedisplay(); 
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
  
  // OpenGL 3.1
  glutInitContextVersion(3,2);
  glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
  
  
  //Create window  
  glutInitWindowSize(window_x,window_y); 
  win = glutCreateWindow("Gfx");
  glutPositionWindow(0,20);

  GLenum GlewInitResult;
  glewExperimental = GL_TRUE;
  GlewInitResult = glewInit();
  
  ignoreErrors();
  printGLError("InitGlew:");

  if (GLEW_OK != GlewInitResult) {
    fprintf(
	    stderr,
	    "ERROR: %s\n",
	    glewGetErrorString(GlewInitResult)
	    );
    exit(EXIT_FAILURE);
  }
  printGLError("before Init:");

  init();

  //Register callbacks 
  glutMouseFunc(mouse);
  glutReshapeFunc(reshape); 
  glutDisplayFunc(display);

  glutIdleFunc(idle);
  glutSpecialFunc(ArrowKeyDown);
  glutSpecialUpFunc(ArrowKeyUp);
  glutTimerFunc(15,timer,0);

 
  
  // print some info
  printf("OpenGL version: %s\n",glGetString(GL_VERSION));
  
  glutMainLoop();
 
  
}   
 
  
 
