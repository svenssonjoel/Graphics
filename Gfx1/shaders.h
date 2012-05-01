#ifndef __SHADERS_H
#define __SHADERS_H

#define GL3_PROTOTYPES
#include "gl3.h"

 /* 
   Allocates memory for a file. 
   caller is responsible for freeing that 
   memory.
 */
extern char* textFileRead(char *fn);

extern GLuint loadShader(char *fn, GLenum kind);
extern void printCompileLog(GLuint shader);
extern void printLinkLog(GLuint program);

#endif
