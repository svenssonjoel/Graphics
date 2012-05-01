// C99 
//


#include <stdio.h>
#include <stdlib.h>


#include <GL/glew.h>
#include <GL/freeglut.h>  

#include "Graphics/mesh.h"
#include "Graphics/matrix.h"
#include "Graphics/shader.h"
#include "Graphics/image.h"
#include "Graphics/error.h"
#include "Graphics/texture.h"

// -----------------------------------------------------------------------------
// Globals.. 
// -----------------------------------------------------------------------------

GLuint p,f,v;  // program, fragshader, vertexshader

GLuint const_p,const_f,const_v; // program for constant color shading

Mat4x4f projectionMatrix; 
Mat4x4f modelViewMatrix; 


GLuint projMatrixIndex;
GLuint modMatrixIndex;

GLuint constProjMatrixIndex;
GLuint constModMatrixIndex;


GLuint tcIndex;
GLuint vertexIndex;
GLuint texLoc1;

GLuint constColorIndex;
GLuint constVertexIndex;

GLuint vertexBufferName;


// GLuint vertBufName;
GLuint quadBufName;
GLuint groundBufName;

GLint  textureID1 = 0; 

float tankPos = 0.0;



// -----------------------------------------------------------------------------
// Triangle data
// -----------------------------------------------------------------------------

const float quad[] = { 
 
  0.0f, 1.0f,  /* texture coords */
  1.0f, 1.0f,  
  1.0f, 0.0f,
  0.0f, 0.0f,
 
 -25.0f,-25.0f,  /*vertex*/
  25.0f,-25.0f, 
  25.0f, 25.0f,
 -25.0f, 25.0f
};

const float ground[] = { 
 
   0.0f, 0.0f,  /*vertex*/
   0.0f, 20.0f, 
  512.0f, 20.0f,
  512.0f, 0.0f
};




void initBuffer(void) {

  glGenBuffers(1, &quadBufName);
  glBindBuffer(GL_ARRAY_BUFFER, quadBufName);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  printGLError("initBuffer:");

  glGenBuffers(1, &groundBufName);
  glBindBuffer(GL_ARRAY_BUFFER, groundBufName);
  glBufferData(GL_ARRAY_BUFFER, sizeof(ground), ground, GL_STATIC_DRAW);
  printGLError("initBuffer:");
 
}


// -----------------------------------------------------------------------------
// setShaders
// -----------------------------------------------------------------------------
void setShaders() {
	
 
	
  v = shader_load("Shaders/tex_simple.vert",GL_VERTEX_SHADER);
  f = shader_load("Shaders/tex_simple.frag",GL_FRAGMENT_SHADER);

  glCompileShader(v);
  shader_printCompileLog(v);
  glCompileShader(f);
  shader_printCompileLog(f);

  p = glCreateProgram();
		
  glAttachShader(p,v);
  glAttachShader(p,f);

  glLinkProgram(p);

  //glUseProgram(p);
  
    
  projMatrixIndex = glGetUniformLocation(p, "proj");
  modMatrixIndex = glGetUniformLocation(p, "mod");

  texLoc1 = glGetUniformLocation(p,"tex"); 
  
  printGLError("setShader3:");
  vertexIndex     = glGetAttribLocation(p, "Vertex");  
  printGLError("setShader4:");
  
  printGLError("setShader5:");



  printGLError("setShader5:");
  tcIndex     = glGetAttribLocation(p, "TexCoord0");  
  printGLError("setShader6:");
  printGLError("setShader7:");

  /* The constant color shader */ 
	
  const_v = shader_load("Shaders/const_mv.vert",GL_VERTEX_SHADER);
  const_f = shader_load("Shaders/const_mv.frag",GL_FRAGMENT_SHADER);

  glCompileShader(const_v);
  shader_printCompileLog(const_v);
  glCompileShader(const_f);
  shader_printCompileLog(const_f);

  const_p = glCreateProgram();
		
  glAttachShader(const_p,const_v);
  glAttachShader(const_p,const_f);

  glLinkProgram(const_p);

  //glUseProgram(const_p);
  
  constProjMatrixIndex = glGetUniformLocation(const_p, "proj");
  constModMatrixIndex  = glGetUniformLocation(const_p, "mod");
    
  constColorIndex   = glGetUniformLocation(const_p, "color");

  constVertexIndex  = glGetAttribLocation(const_p, "Vertex"); 

  fprintf(stderr,"%d\n", constProjMatrixIndex);
  fprintf(stderr,"%d\n", constModMatrixIndex);

  fprintf(stderr,"%d\n", constVertexIndex);
 
  printGLError("constShader:");

}

// -----------------------------------------------------------------------------
// Quad
// -----------------------------------------------------------------------------


void drawQuad(void) {

  glUseProgram(p);
  /* activate texture and connect texture to texture unit */
  glEnableVertexAttribArray(vertexIndex);
  glEnableVertexAttribArray(tcIndex);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureID1);
  glUniform1i(texLoc1,0);


  matrix_identity4x4f(modelViewMatrix);
  matrix_translate4x4f(modelViewMatrix,tankPos,35.0,0);
  
  glUniformMatrix4fv(projMatrixIndex, 1, GL_FALSE, projectionMatrix);
  glUniformMatrix4fv(modMatrixIndex, 1, GL_FALSE,  modelViewMatrix);


  
  glBindBuffer(GL_ARRAY_BUFFER, quadBufName);

  glVertexAttribPointer(tcIndex,2, GL_FLOAT, GL_FALSE,
  			0,(void*)0);

  glVertexAttribPointer(vertexIndex,2, GL_FLOAT, GL_FALSE,
  			0,(void*)32);

 
  
  glDrawArrays(GL_TRIANGLE_FAN,0, 4);
  glDisableVertexAttribArray(vertexIndex);
  glDisableVertexAttribArray(tcIndex);

 
}

void drawGround(void) {

  float color[]= {0.139,0.039,0.019,1.0};
  
  glUseProgram(const_p);
  printGLError("drawGround0:");

  glEnableVertexAttribArray(constVertexIndex);
 

  matrix_identity4x4f(modelViewMatrix);

  glUniformMatrix4fv(constProjMatrixIndex, 1, GL_FALSE, projectionMatrix);
  printGLError("drawGround1a:");
  glUniformMatrix4fv(constModMatrixIndex, 1, GL_FALSE,  modelViewMatrix);
  printGLError("drawGround1b:");
  glUniform4fv(constColorIndex, 1, color);
  
  //printGLError("drawGround1c:");
 
  glBindBuffer(GL_ARRAY_BUFFER, groundBufName);

  printGLError("drawGround2:");
  glVertexAttribPointer(constVertexIndex,2, GL_FLOAT, GL_FALSE,
    			0,(void*)0);
  printGLError("drawGround3:");
 
  
  glDrawArrays(GL_TRIANGLE_FAN,0, 4);

  glDisableVertexAttribArray(constVertexIndex);

}




// -----------------------------------------------------------------------------
// display callback function
// -----------------------------------------------------------------------------
void display(void) { 
  
  glClearColor(1.0,1.0,1.0,1.0);
  glClear(GL_COLOR_BUFFER_BIT);


  
  drawGround();   
  drawQuad();

 


  
  glutSwapBuffers();
  printGLError("display:");

}

// -----------------------------------------------------------------------------
// idle callback function
// -----------------------------------------------------------------------------
void idle(void) {

  glutPostRedisplay();

}

// -----------------------------------------------------------------------------
// reshape
// -----------------------------------------------------------------------------

void reshape(int w, int h) {

  //float ratio = (w <= h) ? (float) h / (float) w : (float) w / (float) h;

  
 
  
  if (w <= h) 
    matrix_ortho2Df(projectionMatrix, 0.0f, w, 0.0f, h);
  else 
    matrix_ortho2Df(projectionMatrix, 0.0f, w, 0.0f, h);

  glViewport(0,0,w,h);

}

/* -----------------------------------------------------------------------------
   keyHandler
   -------------------------------------------------------------------------- */
void keyHandler(int key , int x, int y) {

  switch(key) {
    case GLUT_KEY_LEFT:
      tankPos -= 1;
      break;
    case GLUT_KEY_RIGHT:
      tankPos += 1;
    default:
      break;
  }

}

void mouseHandler(int b, int state, int x, int y) {
  
  if (state == GLUT_DOWN && b == GLUT_LEFT_BUTTON) 
    tankPos -= 0.1;
  else 
    if (state == GLUT_DOWN && b == GLUT_RIGHT_BUTTON) 
      tankPos += 0.1;
}


/* -----------------------------------------------------------------------------
   init (set various things up)
   -------------------------------------------------------------------------- */
void init(void)
{
  
  initBuffer();

  loadTexture2D("Pictures/uglytank.raw",256,256,32,GL_RGBA,textureID1);
  
  
  setShaders(); 

  matrix_identity4x4f(modelViewMatrix);

}


// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
  
  int win;
    
  //Initialize glut
  glutInit(&argc,argv);
  
 
  //Set graphics 
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  
  // OpenGL 3.1
  glutInitContextVersion(3,2);
  glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
  
  
  //Create window 
  glutInitWindowSize(512,512); 
  win = glutCreateWindow("Gfx");
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
  
  error_flushAll(); 

  init();
  
 
  //Register callbacks
  glutReshapeFunc(reshape); 
  glutDisplayFunc(display);
  glutSpecialFunc(keyHandler);
  glutMouseFunc(mouseHandler);
  glutIdleFunc(idle);
  
  // print some info
  printf("OpenGL version: %s\n",glGetString(GL_VERSION));
  
  glutMainLoop();
 
  
}
