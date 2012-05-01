
#include "tools.h"
#include <stdio.h>
#include <stdlib.h>

#include "error.h"
/* -----------------------------------------------------------------------------
   drawLine
   -------------------------------------------------------------------------- */

void tools_drawLine(Shader *s, Vector3f *from, Vector3f *to, Vector3f *color) {

  glUseProgram(s->shader);
 
  printGLError("drawLine0:");
  
  for (int i = 0; i < s->num_uniforms; ++i) {
    switch (s->uniforms[i].type) { 
    case UNIFORM_UNDEFINED:  
      fprintf(stderr,"drawLine: Undefined uniform\n");  
      exit(EXIT_FAILURE);
    case UNIFORM_MAT3X3F:  
      glUniformMatrix3fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m3x3f);
      printGLError("drawLine3 A-2-1:");
      break;
    case UNIFORM_MAT4X4F:
      glUniformMatrix4fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m4x4f);
      printGLError("drawLine3 A-2-2:"); 
      break;  
    case UNIFORM_VEC3F:    
      glUniform3f(s->uniforms[i].id, 
		  s->uniforms[i].data.v3f->x,
		  s->uniforms[i].data.v3f->y,
		  s->uniforms[i].data.v3f->z);
      printGLError("drawLine3 A-2-3:");
      break;
    case UNIFORM_VEC4F:
      glUniform4f(s->uniforms[i].id,
      		  s->uniforms[i].data.v4f->x,
		  s->uniforms[i].data.v4f->y,
		  s->uniforms[i].data.v4f->z,
		  s->uniforms[i].data.v4f->w);
      printGLError("drawLine3 A-2-4:"); 
      break; 
    case UNIFORM_INT: 
      glUniform1i(s->uniforms[i].id, 
		  *s->uniforms[i].data.i);
      printGLError("drawLine3 A-2-5:");
      break; 
    default:  
      /* Silently fall through */
      break;
    }  
  }
  printGLError("drawLine3 A-2:");

  GLuint line; 
  GLuint cols; 
  Vector3f endpoints[2] = 
    {{from->x, from->y, from->z},
     {to->x, to->y, to->z}};

  Vector3f colors[2] = 
    {{color->x,color->y,color->z},
     {color->x,color->y,color->z}};


  glGenBuffers(1, &line);
  glBindBuffer(GL_ARRAY_BUFFER, line);
  glBufferData(GL_ARRAY_BUFFER, 2*sizeof(Vector3f), endpoints, GL_DYNAMIC_DRAW);
  printGLError("drawLine3 A-1:");
  glGenBuffers(1, &cols);
  glBindBuffer(GL_ARRAY_BUFFER, cols);
  glBufferData(GL_ARRAY_BUFFER, 2*sizeof(Vector3f), colors, GL_DYNAMIC_DRAW);
 
  printGLError("drawLine3 A:");
 
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, line);
    glVertexAttribPointer(SHADER_ATTRIB(s,VERTEX_INDEX),
			  3, GL_FLOAT, GL_FALSE,0,(void*)0);
  }
   printGLError("drawLine3 B:");
  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, cols);   
    glVertexAttribPointer(SHADER_ATTRIB(s,COLOR_INDEX),
			  3,GL_FLOAT,GL_FALSE,0,(void*)0);
  }
  printGLError("drawLine3 C:");
 
  printGLError("drawLine3:");
 
 
  glDrawArrays(GL_LINES, 0, 2);
  
  
 
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));

  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));

  glDeleteBuffers(1,&line);
  glDeleteBuffers(1,&cols);
 
}
  
