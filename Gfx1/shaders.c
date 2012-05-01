/* 
   Helper functions to load and compile shaders.
   

*/

#include <stdio.h>
#include <stdlib.h>

#include "shaders.h"



 
// -----------------------------------------------------------------------------
// textFileRead
// -----------------------------------------------------------------------------
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


// -----------------------------------------------------------------------------
// 
// -----------------------------------------------------------------------------

//extern GLuint glCreateShader(GLuint);
//extern void glShaderSource(GLuint,GLsizei,const char**,const GLint*);

GLuint loadShader(char *fn, GLenum kind) {
  GLuint s; 
  char *prg;
  
  
  s = glCreateShader(kind);
  
  prg = textFileRead(fn); 
  
  // fprintf(stderr,"%s",prg);
  
  const char * prg_ = prg;
  
  glShaderSource(s,1,&prg_,NULL);
  
  free(prg);
  return s;
}
  
 
   

// -----------------------------------------------------------------------------
// printCompileLog(GLuint shader) 
// -----------------------------------------------------------------------------
void printCompileLog(GLuint shader) {
  char *log;
  int   length;
  
  glGetShaderiv(shader,GL_INFO_LOG_LENGTH, &length);
  log = (char*)malloc(length);
  glGetShaderInfoLog(shader,length,NULL,log);
  fprintf(stdout,"%s\n",log);
  free(log);

}

// -----------------------------------------------------------------------------
// void printLinkLog(GLuint program);
// -----------------------------------------------------------------------------
void printLinkLog(GLuint prg) {
  char *log;
  int   length;
  
  glGetProgramiv(prg,GL_INFO_LOG_LENGTH, &length);
  log = (char*)malloc(length);
  glGetProgramInfoLog(prg,length,NULL,log);
  fprintf(stdout,"%s\n",log);
  free(log);

}
