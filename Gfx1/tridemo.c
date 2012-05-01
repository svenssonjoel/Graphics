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
Mat4x4 modelViewMatrix; 


GLuint projMatrixIndex;
GLuint modMatrixIndex;


GLuint colorIndex;
GLuint vertexIndex;

GLuint vertexBufferName;


GLuint vertBufName;
GLuint quadBufName;


float rad = 0;
// -----------------------------------------------------------------------------
// Data
// -----------------------------------------------------------------------------

typedef struct {
  float x;
  float y; 
  float z; 
  float dx;
  float dy;
  float dz;
  float rx;
  float ry;
  float rz;
  float rads;
  float radSpeed;
  int active; 
} posDir;

#define NUM_TRI 10
posDir triArray[NUM_TRI]; 

void initTris() {

  for (int i = 0; i < NUM_TRI; i ++) {
    triArray[i].x = 10 + rand() % 20;
    triArray[i].y = 10 + rand() % 20; 
    triArray[i].z = 1; 
    triArray[i].dx = (5 - rand() % 10) / 10000.0f;
    triArray[i].dy = (5 - rand() % 10) / 10000.0f;
    triArray[i].dz = 0;
    triArray[i].rx = 0;
    triArray[i].ry = 0; 
    triArray[i].rz = 1.0;
    triArray[i].rads = 0; 
    triArray[i].radSpeed = 0.0005 - rand() % 1000 / 1000000.0f; 
    triArray[i].active = 1; 
    

  }
}


void updateTris() {

  for (int i = 0; i < NUM_TRI; i ++) {
    if (triArray[i].active) {
      triArray[i].x += triArray[i].dx;
      triArray[i].y += triArray[i].dy;
      triArray[i].z += triArray[i].dz; 
      //triArray[i].rz += 0.001;  //triArray[i].radSpeed;
      triArray[i].rads += triArray[i].radSpeed;

      
      if (triArray[i].x < 0 || triArray[i].x > 40 || 
	  triArray[i].y < 0 || triArray[i].y > 40) {
	triArray[i].active = 0; 
      }
	
      

    }
    else {
      triArray[i].x = 10 + rand() % 20;
      triArray[i].y = 10 + rand() % 20; 
      triArray[i].z = 1; 
      triArray[i].dx = (float)(5 - rand() % 10) / 10000.0f;
      triArray[i].dy = (float)(5 - rand() % 10) / 10000.0f;
      triArray[i].dz = 0;
      triArray[i].rx = 0; 
      triArray[i].ry = 0; 
      triArray[i].rz = 1.0;
      triArray[i].rads = 0; 
      triArray[i].radSpeed = 0.0005 -  (rand() % 1000) / 1000000.0f; 
      triArray[i].active = 1; 
    }
  }
}





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
  -2.5f, 5.0f,       /*vertex*/
  2.5f, 0.0f, 
  -2.5f, -5.0f
};

const float quad[] = { 
  1.0f, 0.0f, 0.0f, /*color*/
  0.0f, 1.0f, 0.0f, 
  0.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 
 -1.0f, 1.0f,       /*vertex*/
  1.0f, 1.0f, 
  1.0f,-1.0f,
 -1.0f,-1.0f
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

  glGenBuffers(1, &quadBufName);
  glBindBuffer(GL_ARRAY_BUFFER, quadBufName);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  printGLError("initBuffer:");


 
}




 

// -----------------------------------------------------------------------------
// setShaders
// -----------------------------------------------------------------------------
void setShaders() {
	
  char *vs,*fs;
	
  v = loadShader("Shaders/flat_mv.vert",GL_VERTEX_SHADER);
  f = loadShader("Shaders/flat_mv.frag",GL_FRAGMENT_SHADER);

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
  modMatrixIndex = glGetUniformLocation(p, "mod");

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
  glUniformMatrix4fv(modMatrixIndex, 1, GL_FALSE,  modelViewMatrix);

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
  glUniformMatrix4fv(modMatrixIndex, 1, GL_FALSE,  modelViewMatrix);

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

void triangles(void) {
  glUniformMatrix4fv(projMatrixIndex, 1, GL_FALSE, projectionMatrix);


  for ( int i = 0; i < NUM_TRI; i ++) {

    if (triArray[i].active) {
      loadIdentity4x4(modelViewMatrix);
    
    
    
      translate4x4(modelViewMatrix,
		   triArray[i].x,
		   triArray[i].y,
		   triArray[i].z);
	
       
       rotate4x4(modelViewMatrix,
		triArray[i].rads,
		triArray[i].rx,
		triArray[i].ry,
		triArray[i].rz);


    
      glUniformMatrix4fv(modMatrixIndex, 1, GL_FALSE,  modelViewMatrix);
  
      glBindBuffer(GL_ARRAY_BUFFER, vertBufName);

      glVertexAttribPointer(colorIndex,3, GL_FLOAT, GL_FALSE,
			    0,(void*)0);
   
      glVertexAttribPointer(vertexIndex,2, GL_FLOAT, GL_FALSE,
			    0,(void*)36);
   
      glDrawArrays(GL_TRIANGLES,0, 3);
    }
  }
  
}


void drawQuad(void) {
  glUniformMatrix4fv(projMatrixIndex, 1, GL_FALSE, projectionMatrix);
  glUniformMatrix4fv(modMatrixIndex, 1, GL_FALSE,  modelViewMatrix);


  
  glBindBuffer(GL_ARRAY_BUFFER, quadBufName);

  glVertexAttribPointer(colorIndex,3, GL_FLOAT, GL_FALSE,
			0,(void*)0);

  glVertexAttribPointer(vertexIndex,2, GL_FLOAT, GL_FALSE,
			0,(void*)48);

  glDrawArrays(GL_TRIANGLE_FAN,0, 4);

  
}




// -----------------------------------------------------------------------------
// display callback function
// -----------------------------------------------------------------------------
void display(void) { 
  
  glClear(GL_COLOR_BUFFER_BIT);
  
  //   triangle_alt();
  
  //drawQuad();
  
  triangles();
  
  glutSwapBuffers();
  printGLError("display:");

}

// -----------------------------------------------------------------------------
// idle callback function
// -----------------------------------------------------------------------------
void idle(void) {
  
  updateTris();
  
  glutPostRedisplay();
  
  //loadIdentity4x4(modelViewMatrix);
  //translate4x4(modelViewMatrix,15,15,0);
  //rotate4x4(modelViewMatrix,rad,0,0,1);
  
  
  //rad += 0.0001;
}

// -----------------------------------------------------------------------------
// reshape
// -----------------------------------------------------------------------------

void reshape(int w, int h) {

  float ratio = (w <= h) ? (float) h / (float) w : (float) w / (float) h;

  
 
  
  if (w <= h) 
    setOrtho2D(projectionMatrix, 0.0f, 30.0f, 0.0f, 30.0f * ratio);
  else 
    setOrtho2D(projectionMatrix, 0.0f, 30.0f * ratio, 0.0f, 30.0f);

   glViewport(0,0,w,h);

}



// -----------------------------------------------------------------------------
// init (set various things up)
// -----------------------------------------------------------------------------
void init(void)
{

  initBuffer();
  setShaders();

  initTris();

  loadIdentity4x4(modelViewMatrix);
  //scale4x4(modelViewMatrix,1.5,1.5,1);
 
 
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
  glutInitWindowSize(512,512); 
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
