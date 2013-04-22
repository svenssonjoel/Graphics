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
float py = 0.0;

Vector4f earth_pos = {0.0,0.0,10.0,1.0};
Vector4f sun_pos = {0.0,0.0,0.0,1.0}; 

// -----------------------------------------------------------------------------
// Globals.. 
// -----------------------------------------------------------------------------
int win;

//GLuint p,f,v;  // program, fragshader, vertexshader



/* Matrices, Projection, modelview and normal */
Mat4x4f projectionMatrix; 
Mat4x4f modelViewMatrix; 
Mat3x3f normalMatrix;  


GLint texture1  = 1;
GLint texture2  = 2;

int window_x = 800; 
int window_y = 600;

float rotation = 0.0;
Vector4f lightsource;

//Image *heightmap;

// -----------------------------------------------------------------------------
// Triangle data
// -----------------------------------------------------------------------------

//#define  SPHERE_DETAIL 10
#define  SPHERE_DETAIL 100

Mesh *sphere_mesh;
Uniform sphere_unis[3];
Uniform ground_unis[3];
GLint texunit;
Vector4f mesh_color = {1.0,0.0,0.0,1.0};

Shader shader1; 
Shader shader2; 


// -----------------------------------------------------------------------------
// earth and sun data (mesh, shader, texture)
// -----------------------------------------------------------------------------

Mesh *earth_mesh;
Uniform earth_shader_unis[5];
GLint earth_texture;
Shader earth_shader;

Mesh *sun_mesh;
Uniform sun_shader_unis[3];
GLint sun_texture;
Shader sun_shader;





// -----------------------------------------------------------------------------
// setShaders
// -----------------------------------------------------------------------------


void setShaders() {
	
  printGLError("before:");       
  
  GLuint shader_frag;
  GLuint shader_vert; 
  GLuint shader_program;



  GLuint shader_vindex;
  GLuint shader_tindex;
  GLuint shader_nindex;
  GLuint shader_proj;
  GLuint shader_mod;
  GLuint shader_normal_matrix;
  GLuint shader_texSampler;
  GLuint shader_lightpos;


  /* setup texturing shader ---------------------------------------------------------------------- */

 
  shader_vert = shader_load("Shaders/earth_mv.vert",
				 GL_VERTEX_SHADER);
  printGLError("setShaderD: earth");
  shader_frag = shader_load("Shaders/earth_mv.frag",
	 			 GL_FRAGMENT_SHADER);

  printGLError("setShaderC: earth");

  glCompileShader(shader_vert);
  shader_printCompileLog(shader_vert);
  glCompileShader(shader_frag);
  shader_printCompileLog(shader_frag);
  printGLError("setShaderB: earth");

   
  shader_program = glCreateProgram();
  printGLError("setShaderA: earth");

		
  glAttachShader(shader_program,shader_vert);
  glAttachShader(shader_program,shader_frag);
  printGLError("setShader8: earth");

  glLinkProgram(shader_program);
  
 

  shader_proj = glGetUniformLocation(shader_program, "proj");
  shader_mod = glGetUniformLocation(shader_program, "mod");
  shader_texSampler = glGetUniformLocation(shader_program, "tex");
  shader_normal_matrix = glGetUniformLocation(shader_program,"normalMatrix");
  shader_lightpos = glGetUniformLocation(shader_program,"lightPos");

  
  printGLError("setShader9:");
  shader_vindex = glGetAttribLocation(shader_program, "Vertex");
  shader_tindex = glGetAttribLocation(shader_program, "TexCoord0");
  shader_nindex = glGetAttribLocation(shader_program, "Normal");

 
  /* setup shader for earth_mesh */
  
  earth_shader_unis[0].type = UNIFORM_MAT4X4F;
  earth_shader_unis[0].id   = shader_proj;
  earth_shader_unis[0].data.m4x4f = &projectionMatrix;
  
  earth_shader_unis[1].type = UNIFORM_MAT4X4F;
  earth_shader_unis[1].id   = shader_mod;
  earth_shader_unis[1].data.m4x4f = &modelViewMatrix;
 
  earth_shader_unis[2].type = UNIFORM_MAT3X3F;
  earth_shader_unis[2].id   = shader_normal_matrix;
  earth_shader_unis[2].data.m3x3f = &normalMatrix;
 
  earth_shader_unis[3].type = UNIFORM_INT;
  earth_shader_unis[3].id   = shader_texSampler;
  earth_shader_unis[3].data.i = &texunit;

  earth_shader_unis[4].type = UNIFORM_VEC3F;
  earth_shader_unis[4].id   = shader_lightpos;
  earth_shader_unis[4].data.v3f = VECTOR_SUB3F(&lightsource);

 
  earth_shader.uniforms = earth_shader_unis; 
  earth_shader.num_uniforms = 5;
  earth_shader.shader   = shader_program;
  earth_shader.attributes[VERTEX_INDEX].vattrib = shader_vindex;
  earth_shader.attributes[VERTEX_INDEX].active = true;
  earth_shader.attributes[TEXEL_INDEX].vattrib = shader_tindex;
  earth_shader.attributes[TEXEL_INDEX].active = true;
  earth_shader.attributes[NORMAL_INDEX].vattrib = shader_nindex;
  earth_shader.attributes[NORMAL_INDEX].active = true;
  
/* -----------------------------------------------------------------------------
   Sun shaders
   -------------------------------------------------------------------------- */
  
  shader_vert = shader_load("Shaders/sun_mv.vert",  
				 GL_VERTEX_SHADER);
  shader_frag = shader_load("Shaders/sun_mv.frag",
				 GL_FRAGMENT_SHADER);


  glCompileShader(shader_vert);
  shader_printCompileLog(shader_vert);
  glCompileShader(shader_frag);
  shader_printCompileLog(shader_frag);
   
  shader_program = glCreateProgram(); 

		
  glAttachShader(shader_program,shader_vert);
  glAttachShader(shader_program,shader_frag);
  printGLError("setShader8:");

  glLinkProgram(shader_program);
  
 

  shader_proj = glGetUniformLocation(shader_program, "proj");
  shader_mod = glGetUniformLocation(shader_program, "mod");
  shader_texSampler = glGetUniformLocation(shader_program, "tex");

  printGLError("setShader9:");
  shader_vindex = glGetAttribLocation(shader_program, "Vertex");
  shader_tindex = glGetAttribLocation(shader_program, "TexCoord0");

  sun_shader_unis[0].type = UNIFORM_MAT4X4F;
  sun_shader_unis[0].id   = shader_proj;
  sun_shader_unis[0].data.m4x4f = &projectionMatrix;
  
  sun_shader_unis[1].type = UNIFORM_MAT4X4F;
  sun_shader_unis[1].id   = shader_mod;
  sun_shader_unis[1].data.m4x4f = &modelViewMatrix;
 
  sun_shader_unis[2].type = UNIFORM_INT;
  sun_shader_unis[2].id   = shader_texSampler;
  sun_shader_unis[2].data.i = &texunit;

  sun_shader.uniforms = sun_shader_unis; 
  sun_shader.num_uniforms = 3;
  sun_shader.shader   = shader_program;
  sun_shader.attributes[VERTEX_INDEX].vattrib = shader_vindex;
  sun_shader.attributes[VERTEX_INDEX].active = true;
  sun_shader.attributes[TEXEL_INDEX].vattrib = shader_tindex;
  sun_shader.attributes[TEXEL_INDEX].active = true;
 
}
// -----------------------------------------------------------------------------
// drawLine
// -----------------------------------------------------------------------------
/*
void drawLine(Vector3f *from, Vector3f *to, Vector3f *color) {

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



  printGLError("drawLine3:");
 
 
  glDrawArrays(GL_LINES, 0, 2);
  
  


  glDisableVertexAttribArray(shader1_vindex);
 
}
*/

    
// -----------------------------------------------------------------------------
// display callback function
// -----------------------------------------------------------------------------
void display(void) {    
 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  matrix_identity4x4f(modelViewMatrix);

  /* SET UP VIEW */
  Vector3f pos = vector3f(px,py,pz);
  Vector3f la  = vector3f(px+sin(3.14-r),py,pz+cos(3.14-r));
  Vector3f up  = vector3f(0.0,1.0,0.0);
  matrix_lookAtf(modelViewMatrix,&pos, &la  ,&up);
  
  /* POSITION THE SUN */
  matrix_translate4x4f(modelViewMatrix,sun_pos.x,sun_pos.y,sun_pos.z);
  
  /* RENDER THE SUN */
  mesh_render(&sun_shader,sun_mesh);
 

  /* TRANSFORM THE LIGHTSOURCE INTO CAMERACOORDINATES*/
  matrix_transform4x4f(&lightsource,modelViewMatrix,&sun_pos);

  /* SET UP EARTH ROTATION */
  matrix_rotate4x4f(modelViewMatrix,-rotation, 0.0,1.0,0.0);
  matrix_translate4x4f(modelViewMatrix,earth_pos.x,earth_pos.y,earth_pos.z);
  matrix_rotate4x4f(modelViewMatrix,-2.14/2, 0.0,0.0,1.0);
  matrix_rotate4x4f(modelViewMatrix,rotation, 1.0,0.0,0.0);
 

  
  /* SET UP NORMAL TRANSFORMATION MATRIX */
  matrix_sub3x3f(normalMatrix,modelViewMatrix);
  matrix_invert3x3f(normalMatrix,normalMatrix);
  matrix_transpose3x3f(normalMatrix,normalMatrix);
  
  /* RENDER EARTH */
  mesh_render(&earth_shader,earth_mesh);

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
   
  Image *img; 
  
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK); 
  glEnable(GL_DEPTH_TEST);


  earth_mesh = mesh_create();
  sun_mesh = mesh_create();


  /* EARTH */ 

  earth_mesh->vertices = (Vector3f *)malloc(SHAPES_SPHERE_VERTICES(SPHERE_DETAIL)*sizeof(Vector3f));
  earth_mesh->texels = (Vector2f *)malloc(SHAPES_SPHERE_VERTICES(SPHERE_DETAIL)*sizeof(Vector2f));  
  earth_mesh->normals = (Vector3f *)malloc(SHAPES_SPHERE_VERTICES(SPHERE_DETAIL)*sizeof(Vector3f));
  earth_mesh->texture_id = texture1;
  earth_mesh->textured = true;
  earth_mesh->num_vertices = SHAPES_SPHERE_VERTICES(SPHERE_DETAIL);
  earth_mesh->indices = (unsigned char*)malloc(SHAPES_SPHERE_INDICES(SPHERE_DETAIL)*sizeof(unsigned int));
  earth_mesh->num_indices = SHAPES_SPHERE_INDICES(SPHERE_DETAIL);
  earth_mesh->indices_type = GL_UNSIGNED_INT;
  
  /* generate the sphere data */
  shapes_sphere(earth_mesh->vertices,
		earth_mesh->normals,
		earth_mesh->texels,
		(unsigned int*)earth_mesh->indices,1.0,SPHERE_DETAIL);
  
  
  mesh_upload(earth_mesh);    

  /* SUN */ 

  sun_mesh->vertices = (Vector3f *)malloc(SHAPES_SPHERE_VERTICES(SPHERE_DETAIL)*sizeof(Vector3f));
  sun_mesh->texels = (Vector2f *)malloc(SHAPES_SPHERE_VERTICES(SPHERE_DETAIL)*sizeof(Vector2f));  
  sun_mesh->normals = (Vector3f *)malloc(SHAPES_SPHERE_VERTICES(SPHERE_DETAIL)*sizeof(Vector3f));
  sun_mesh->texture_id = texture2;
  sun_mesh->textured = true;
  sun_mesh->num_vertices = SHAPES_SPHERE_VERTICES(SPHERE_DETAIL);
  sun_mesh->indices = (unsigned char*)malloc(SHAPES_SPHERE_INDICES(SPHERE_DETAIL)*sizeof(unsigned int));
  sun_mesh->num_indices = SHAPES_SPHERE_INDICES(SPHERE_DETAIL);
  sun_mesh->indices_type = GL_UNSIGNED_INT;
  
  /* generate the sphere data */
  shapes_sphere(sun_mesh->vertices,
		sun_mesh->normals,
		sun_mesh->texels,
		(unsigned int*)sun_mesh->indices,1.0,SPHERE_DETAIL);
  
  
  mesh_upload(sun_mesh);    





  
    
  setShaders(); 
  
  img = image_loadPNG("Pictures/earth1.png");
  image_hFlip(img); 
  
  initTexture2D(texture1,img->width,img->height,GL_RGB,img->data);
  image_destroy(img);

  img = image_loadPNG("Pictures/sun1.png");
  image_hFlip(img); 
  
  initTexture2D(texture2,img->width,img->height,GL_RGB,img->data);
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
    px -= sin(3.14-r);
    pz -= cos(3.14-r);
  }
  if (buttons[UP]) {
    px += sin(3.14-r);
    pz += cos(3.14-r);
  }

  rotation += 0.005;
  

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
  glutInitContextVersion(3,0);
  glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
  glutInitContextProfile(GLUT_CORE_PROFILE);
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

  glutSpecialFunc(ArrowKeyDown);
  glutSpecialUpFunc(ArrowKeyUp);
  glutTimerFunc(10,timer,0);

 
  
  // print some info
  printf("OpenGL version: %s\n",glGetString(GL_VERSION));
  
  glutMainLoop();
 
  
}
