

#include "error.h"

#include <stdio.h>
#include <stdlib.h>


#include <GL/glu.h>

// -----------------------------------------------------------------------------
// printGLError
// -----------------------------------------------------------------------------

void printGLError(char *prefix) {
  int error = glGetError();
  if (error) {
    fprintf(stderr,"%s",prefix);
    fprintf(stderr," %s\n",gluErrorString(error));
    exit(-1);
  }
}
