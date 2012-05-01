

#include "error.h"

#include <stdio.h>
#include <stdlib.h>


#include <GL/glu.h>

// -----------------------------------------------------------------------------
// printGLError
// -----------------------------------------------------------------------------

void ignoreErrors() { error_flushAll(); }

void error_flushAll() {

  while (glGetError());
}

void printGLError(char *prefix) { error_print(prefix);} 
void error_print(char *prefix) {
  int error = glGetError();
  if (error) {
    fprintf(stderr,"%s",prefix);
    fprintf(stderr," %s\n",gluErrorString(error));
    exit(-1);
  }
}
