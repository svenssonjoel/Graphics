// C99 
// 
// gcc -std=c99 main.c vector.c -o main -lglut -lGLEW -msse3

#include <stdio.h>
#include <stdlib.h>


// #include <GL/glew.h> 


#include <GL/glew.h> 
#include <GL/freeglut.h>  /* freeglut version 2.6.0 */




#include "sse_detect.h" 
#include "diff_ms.h" 

#include "mesh.h"
#include "cube.h"

GLfloat light_diffuse[] = {1.0, 0.0, 0.0, 1.0};  /* Red diffuse light. */
GLfloat light_position[] = {1.0, 0.0, 1.0, 0.0};  /* Infinite light location. */


void renderBitmapString(float x, float y, float z, void *font, char *string);  

struct timeval t_start,t_stop;
int frame = 0; 
volatile int fps = 0;
float rotation = 0; 
int show_normals = 1; 


GLuint p,f,v;




/* -------------------------------------------------------------------------------------*/
void printGLError(char *prefix) {
  fprintf(stderr,"%s",prefix);
  fprintf(stderr," %s\n",gluErrorString(glGetError()));
}

/* -------------------------------------------------------------------------------------*/
char *textFileRead(char *fn) {
  FILE *fp;
  char *content = NULL;

  int count=0;

  if (fn != NULL) {

    fp = fopen(fn,"rt");

    if (fp != NULL) {
										      
      fseek(fp, 0, SEEK_END);
      count = ftell(fp);
      rewind(fp);

      if (count > 0) {
	content = (char *)malloc(sizeof(char) * (count+1));
	count = fread(content,sizeof(char),count,fp);
	content[count] = '\0';
      }
      fclose(fp);
										
    }
  }
	
  return content;
}

GLint getUniLoc(GLuint program, const char *name){
 return  glGetUniformLocation(program,name);
}

void setShaders() {
	
  char *vs,*fs;
	
  v  = glCreateShader(GL_VERTEX_SHADER);
  f  = glCreateShader(GL_FRAGMENT_SHADER);	
	
  vs = textFileRead("Shaders/basic_v.glsl");
  fs = textFileRead("Shaders/basic_f.glsl");
	
  const char * vv = vs;
  const char * ff = fs;
	
  glShaderSource(v,1, &vv,NULL);
  glShaderSource(f,1, &ff,NULL);
  printGLError("setShader:");	
  free(vs);free(fs);
	
  glCompileShader(v);
  glCompileShader(f);
  printGLError("setShader:");
  p = glCreateProgram();
		
  glAttachShader(p,v);
  glAttachShader(p,f);
  printGLError("setShader:");
  glLinkProgram(p);
  printGLError("setShader:");
 
  glUseProgram(p);
  
  printGLError("setShader:");
  
  glUniform4f(getUniLoc(p, "Color"), 0.0, 0.3, 0.2, 1.0);
  
 
}


/* -------------------------------------------------------------------------------------*/


void display(void) { 
  char fps_string[128]; 
  frame ++; 
  
  gettimeofday( &t_stop, 0);
  int diff = diff_ms(&t_start,&t_stop);
  if (diff  > 1000) { 
    fps = frame * 1000.0/diff;
    t_start = t_stop;
    frame = 0; 
  }
 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

 
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
 
 
  glTranslatef(0.0, 0.0, -6.0);
  glRotatef(rotation += 0.1,1.0,1.0,1.0);

  glBegin(GL_TRIANGLES);
  for (int i = 0; i < 36; i++) {
   
    glNormal3fv((float *)&cube_n2[i]);
    glVertex3fv((float *)&cube_v2[i]);
    
  }
  glEnd();
  if (show_normals) {
    glBegin(GL_LINES);
      for (int i = 0; i < 36; i++) {
	glVertex3fv((float *)&cube_v2[i]);
	glVertex3f(cube_v2[i].x + cube_n2[i].x,
		   cube_v2[i].y + cube_n2[i].y,
		   cube_v2[i].z + cube_n2[i].z);	
      }
      glEnd();
  }
  
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0, 100, 0, 100);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
 
  snprintf(fps_string,128,"FPS: %d",fps);
  
  renderBitmapString(0,0,0,GLUT_BITMAP_8_BY_13,fps_string);
  
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopAttrib();


 

  glutSwapBuffers();
 

}

void idle(void) {
  
  glutPostRedisplay();
 
}

void renderBitmapString(
		float x, 
		float y, 
		float z, 
		void *font, 
		char *string) {  
  char *c;
  glRasterPos3f(x, y,z);
  for (c=string; *c != '\0'; c++) {
    glutBitmapCharacter(font, *c);
  }
}
void init(void)
{
  printf("%s\n",glGetString(GL_VERSION));
 
  setShaders();


  /* Enable a single OpenGL light. */
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);

  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
		  /* aspect ratio */ 640.0/480.0,
		  /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  
  //gluLookAt(0.0, 0.0, 5.0,      /* eye is at (0,0,5) */
  //	    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
  //	    0.0, 1.0, 0.);      /* up is in positive Y direction */


  //glTranslatef(0.0, 0.0, -1.0);
  //glRotatef(60, 1.0, 0.0, 0.0);
  //glRotatef(-20, 0.0, 0.0, 1.0);


}

int main(int argc, char **argv) {
  
  int win;
  
  int v = 0;
  int m = 0; 
   
  
  gettimeofday( &t_start, 0);

  sse_version(&v,&m);
  printf("SSE %d.%d supported\n",v,m);
 
   
  
  //Initialize 
  glutInit(&argc,argv);
  
 
  //Set graphics 
  glutInitDisplayMode(GLUT_DEPTH | GLUT_RGB | GLUT_DOUBLE);
  
  
  //Create window 
  glutInitWindowSize(640,480); 
  win = glutCreateWindow("Gfx");


  //Register callbacks
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  


  glewInit();
 
  if (glewIsSupported("GL_VERSION_2_0")) printf("2.0\n");
  if (glewIsSupported("GL_VERSION_2_1")) printf("2.1\n");
 
  init();
  
  
  glutMainLoop();

  
}
