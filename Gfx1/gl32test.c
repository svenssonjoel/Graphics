// C99 
// OpenGL 3.2
//
// make gl32test

#include <stdio.h>
#include <stdlib.h>


// uses a slightly modified version of gl3.h from khronos (think it conflicts with freeglut.h)
#define GL3_PROTOTYPES
#include "gl3.h"

#include <GL/glu.h>

#include <GL/freeglut.h>  /* freeglut version 2.6.0 */

#include "mesh.h"
#include "cube.h"
#include "matrix.h"
#include "shaders.h"

#include "error.h"

// -----------------------------------------------------------------------------
// Globals.. 
// -----------------------------------------------------------------------------

GLuint p,f,v;  // program, fragshader, vertexshader

Mat4x4 projectionMatrix; 

GLuint projMatrixIndex;
GLuint colorIndex;
GLuint vertexIndex;

GLuint vertexBufferName;


GLuint vertBufName;
GLuint colBufName;



// -----------------------------------------------------------------------------
// Triangle data
// -----------------------------------------------------------------------------
int stride = 5*sizeof(float);
int numColorComponents = 3; 
int numVertexComponents = 2;

const float varray[] = { 
  1.0f, 0.0f, 0.0f, /*color*/
  5.0f, 5.0f,       /*vertex*/
  0.0f, 1.0f, 0.0f, 
  25.0f, 5.0f, 
  0.0f, 0.0f, 1.0f,
  5.0f, 25.0f
};



const float Arr[] = { 
  1.0f, 0.0f, 0.0f, /*color*/
  0.0f, 1.0f, 0.0f, 
  0.0f, 0.0f, 1.0f,
  5.0f, 5.0f,       /*vertex*/
  25.0f, 5.0f, 
  5.0f, 25.0f
};


void initBuffer(void) {
  glGenBuffers(1, &vertexBufferName);
  glBindBuffer(GL_ARRAY_BUFFER, vertexBufferName);
  glBufferData(GL_ARRAY_BUFFER, sizeof(varray), varray, GL_STATIC_DRAW);
  printGLError("initBuffer:");


  glGenBuffers(1, &vertBufName);
  glBindBuffer(GL_ARRAY_BUFFER, vertBufName);
  glBufferData(GL_ARRAY_BUFFER, sizeof(Arr), Arr, GL_STATIC_DRAW);
  printGLError("initBuffer:");


 
}


 

// -----------------------------------------------------------------------------
// setShaders
// -----------------------------------------------------------------------------
void setShaders() {
	
  char *vs,*fs;
	
  v = loadShader("Shaders/smooth.vert",GL_VERTEX_SHADER);
  f = loadShader("Shaders/smooth.frag",GL_FRAGMENT_SHADER);

  glCompileShader(v);
  //printCompileLog(v);
  glCompileShader(f);
  //printCompileLog(f);

  p = glCreateProgram();
		
  glAttachShader(p,v);
  glAttachShader(p,f);

  glLinkProgram(p);
  //printLinkLog(p);

  glUseProgram(p);
  
    

  projMatrixIndex = glGetUniformLocation(p, "proj");
  printGLError("setShader3:");
  vertexIndex     = glGetAttribLocation(p, "Vertex");  
  printGLError("setShader4:");
  glEnableVertexAttribArray(vertexIndex);
  printGLError("setShader5:");
  colorIndex      = glGetAttribLocation(p, "Color");

  printGLError("setShader6:");
  glEnableVertexAttribArray(colorIndex);
  printGLError("setShader7:");

  
  
 
}

// -----------------------------------------------------------------------------
// Triangle
// -----------------------------------------------------------------------------


void triangle(void) {
  glUniformMatrix4fv(projMatrixIndex, 1, GL_FALSE, projectionMatrix);
  printGLError("triangle0:");
  
  glBindBuffer(GL_ARRAY_BUFFER, vertexBufferName);
  printGLError("triangle1:");
  glVertexAttribPointer(colorIndex,3, GL_FLOAT, GL_FALSE,
			20,(void*)0);
  printGLError("triangle2:");
  glVertexAttribPointer(vertexIndex,2, GL_FLOAT, GL_FALSE,
			20,(void*)12);
  printGLError("triangle3:");
  glDrawArrays(GL_TRIANGLES,0, 3);
  printGLError("triangle4:");
  
}

void triangle_alt(void) {
  glUniformMatrix4fv(projMatrixIndex, 1, GL_FALSE, projectionMatrix);
  printGLError("triangle0:");
  
  glBindBuffer(GL_ARRAY_BUFFER, vertBufName);
  printGLError("triangle1:");
  glVertexAttribPointer(colorIndex,3, GL_FLOAT, GL_FALSE,
			0,(void*)0);
  printGLError("triangle2:");
  glVertexAttribPointer(vertexIndex,2, GL_FLOAT, GL_FALSE,
			0,(void*)36);
  printGLError("triangle3:");
  glDrawArrays(GL_TRIANGLES,0, 3);
  printGLError("triangle4:");
  
}




// -----------------------------------------------------------------------------
// display callback function
// -----------------------------------------------------------------------------
void display(void) { 
  
  triangle_alt();
  
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

  float ratio = (w <= h) ? (float) h / (float) w : (float) w / (float) h;

  
  glViewport(0,0,w,h);
  
  
  if (w <= h) 
    setOrtho2D(projectionMatrix, 0.0f, 30.0f, 0.0f, 30.0f * ratio);
  else 
    setOrtho2D(projectionMatrix, 0.0f, 30.0f * ratio, 0.0f, 30.0f);


}



// -----------------------------------------------------------------------------
// init (set various things up)
// -----------------------------------------------------------------------------
void init(void)
{

  initBuffer();
  setShaders();

}


// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
  
  int win;
  
  int v = 0;
  int m = 0; 
   
 
  //Initialize glut
  glutInit(&argc,argv);
  
 
  //Set graphics 
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

  // OpenGL 3.1
  glutInitContextVersion(3,2);
  glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
  
  
  //Create window 
  glutInitWindowSize(640,480); 
  win = glutCreateWindow("Gfx");
  
  init();
 
 
  //Register callbacks
  glutReshapeFunc(reshape); 
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  
  // print some info
  printf("OpenGL version: %s\n",glGetString(GL_VERSION));
  
  glutMainLoop();

  
}
