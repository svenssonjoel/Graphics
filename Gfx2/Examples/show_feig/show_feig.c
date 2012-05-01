// C99 

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

Mat4x4f projectionMatrix; 
Mat4x4f modelViewMatrix; 


GLuint projMatrixIndex;
GLuint modMatrixIndex;


GLuint tcIndex;
GLuint vertexIndex;
GLuint texLoc1;
GLuint texLoc2;

GLuint vertexBufferName;

GLuint vertBufName;
GLuint quadBufName;

GLint  textureID1 = 0; 


int window_x; 
int window_y;

// -----------------------------------------------------------------------------
// FEIGENBAUM
// -----------------------------------------------------------------------------

#define  WIDTH  512
#define  HEIGHT 512

#define  LEFT   0.3f
#define  RIGHT  2.9f
#define  BOTTOM 0.0f
#define  TOP    2.0f


#define INVISIBLE 50 
#define VISIBLE   1000


void setUniversalPoint(Image *img, float xu, float yu, unsigned char *color) {
  
  float xs,ys;
  
  xs = (xu - LEFT) * WIDTH / (RIGHT - LEFT);
  ys = (yu - BOTTOM) * HEIGHT / (TOP - BOTTOM); 


 
  /* points are inside the "viewing area" ? */
  if ((unsigned int) xs >= 0 && (unsigned int) xs < WIDTH &&
      (unsigned int) ys >= 0 && (unsigned int) ys < HEIGHT) {
    image_getPixel(img, (unsigned int)xs, (unsigned int)ys,color);
    color[0] = color[0]+5 < 255 ? color[0]+5 : 255; 
    color[1] = color[1]+5 < 255 ? color[1]+5 : 255; 
    color[2] = color[2]+5 < 255 ? color[2]+5 : 255; 
    image_setPixel(img, (unsigned int)xs, (unsigned int)ys,color);
  }
}

float g(float p, float k) {
  return (p + k * p * (1.0f - p));
}


void measlesIteration(Image *img) {
  int range;
  float pop;
  float deltaXPerPixel;
  float feedback;

  unsigned char color[3] = {0,0,0};

  deltaXPerPixel = (RIGHT - LEFT) / (float)WIDTH; 
  
  for (range = 0; range < WIDTH; ++range){

   
    feedback = LEFT + range * deltaXPerPixel;
    pop = 0.3;

    for (int i = 0; i < INVISIBLE; ++i) {
      pop = g(pop,feedback);
    }
    for (int i = 0; i < VISIBLE; ++i) {
      setUniversalPoint(img,feedback,pop,color);
      pop = g(pop,feedback);
    }
    
  }

}



// -----------------------------------------------------------------------------
// Triangle data
// -----------------------------------------------------------------------------

float quad[16];


void windowCoveringQuad(float *data, int w,int h) {
  
  /* window covering quad with up/down inverted  
     so that 0,0 is top left (concerning the texture)
   */ 
  
  
  data[0] = 0.0f; /* texture coords */
  data[1] = 0.0f;  
  data[2] = 1.0f;
  data[3] = 0.0f;
  data[4] = 1.0f;
  data[5] = 1.0f;
  data[6] = 0.0f;
  data[7] = 1.0f;
  data[8] =  0.0f;  /*vertex*/
  data[9] =  0.0f; 
  data[10] = w;
  data[11] = 0.0f;
  data[12] = w;
  data[13] = h;
  data[14] = 0.0f;
  data[15] = h;

  /* 
     (0,0)           (w,0)


     (0,h)           (w,h)

   */ 



  
}

/* Create a window covereing quad and 
   initialize buffers. 
   
   The buffers here are "GL_ARRAY_BUFFERS" 
   which means .... what ??? 
    #is static the correct choice ? 
    #
*/
void initBuffer(void) {

  windowCoveringQuad(quad,window_x,window_y);

  glGenBuffers(1, &quadBufName);
  glBindBuffer(GL_ARRAY_BUFFER, quadBufName);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  printGLError("initBuffer:");
 
}
 
// -----------------------------------------------------------------------------
// setShaders
// -----------------------------------------------------------------------------
void setShaders() {
		
  v = shader_load("Shaders/show_image_mv.vert",GL_VERTEX_SHADER);
  f = shader_load("Shaders/show_image_mv.frag",GL_FRAGMENT_SHADER);

  glCompileShader(v);
  shader_printCompileLog(v);
  glCompileShader(f);
  shader_printCompileLog(f);

  p = glCreateProgram();
		
  glAttachShader(p,v);
  glAttachShader(p,f);

  glLinkProgram(p);
  //printLinkLog(p);

  glUseProgram(p);
  
    
  projMatrixIndex = glGetUniformLocation(p, "proj");
  modMatrixIndex = glGetUniformLocation(p, "mod");

  texLoc1 = glGetUniformLocation(p,"tex");
 
   
  printGLError("setShader3:");
  vertexIndex     = glGetAttribLocation(p, "Vertex");  
  printGLError("setShader4:");
  glEnableVertexAttribArray(vertexIndex);
  printGLError("setShader5:");



  printGLError("setShader5:");
  tcIndex     = glGetAttribLocation(p, "TexCoord0");  
  printGLError("setShader6:");
  glEnableVertexAttribArray(tcIndex);
  printGLError("setShader7:");


}

// -----------------------------------------------------------------------------
// Quad
// -----------------------------------------------------------------------------


void drawQuad(void) {

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureID1);
  glUniform1i(texLoc1,0);

  
  glUniformMatrix4fv(projMatrixIndex, 1, GL_FALSE, projectionMatrix);
  glUniformMatrix4fv(modMatrixIndex, 1, GL_FALSE,  modelViewMatrix);

  
  
  glBindBuffer(GL_ARRAY_BUFFER, quadBufName);

  glVertexAttribPointer(tcIndex,2, GL_FLOAT, GL_FALSE,
			0,(void*)0);

  glVertexAttribPointer(vertexIndex,2, GL_FLOAT, GL_FALSE,
			0,(void*)32);
  
  glDrawArrays(GL_TRIANGLE_FAN,0, 4);

  
}



// -----------------------------------------------------------------------------
// display callback function
// -----------------------------------------------------------------------------
void display(void) { 
  
  glClear(GL_COLOR_BUFFER_BIT);
 
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
  window_x = w;
  window_y = h;

  matrix_ortho2Df(projectionMatrix, 0.0f, (float)w, 0.0f, (float)h);
  
  initBuffer();
  
  glViewport(0,0,w,h);

}



// -----------------------------------------------------------------------------
// init (set various things up)
// -----------------------------------------------------------------------------
void init(void)
{
  
  initBuffer();

 
  Image *out; 

  out = image_create(WIDTH,HEIGHT,3);
  image_clear(out);

  /* create feigenbaum image */
  measlesIteration(out);

  //image_gridTemplate(out,32,32);

  /* store image to texture */
  initTexture2D(textureID1,WIDTH,HEIGHT,GL_RGB,out->data);  

  image_destroy(out);


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
  window_x = window_y = 512;
  glutInitWindowSize(window_x,window_y); 
  win = glutCreateWindow("Gfx");

  GLenum GlewInitResult;
  glewExperimental = GL_TRUE;
  GlewInitResult = glewInit();
  
  error_flushAll();
 
  if (GLEW_OK != GlewInitResult) {
    fprintf(
	    stderr,
	    "ERROR: %s\n",
	    glewGetErrorString(GlewInitResult)
	    );
    exit(EXIT_FAILURE);
  }

  init();
  
 
  //Register callbacks
  glutReshapeFunc(reshape); 
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  
  // print some info
  printf("OpenGL version: %s\n",glGetString(GL_VERSION));
  
  glutMainLoop();
 
  
}
