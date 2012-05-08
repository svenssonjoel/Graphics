
#ifndef __SHADER_H_
#define __SHADER_H_

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "vector.h"
#include "matrix.h"

/* -----------------------------------------------------------------------------
   UNIFORMS (a shader concept)
   -------------------------------------------------------------------------- */


// Uniform types. 
#define UNIFORM_UNDEFINED   0
#define UNIFORM_MAT3X3F     1
#define UNIFORM_MAT4X4F     2
#define UNIFORM_VEC3F       3 
#define UNIFORM_VEC4F       4 
#define UNIFORM_VEC3F_ARRAY 5
#define UNIFORM_VEC4F_ARRAY 6
#define UNIFORM_INT         7
#define UNIFORM_FLOAT       8
#define UNIFORM_UINT        9


typedef struct {
  unsigned int type;
  GLuint   id;
  union {
    Mat4x4f   *m4x4f; // replace all of these with a "uchar *"
    Mat3x3f   *m3x3f; // more casting required then I guess (in the application code) . 
    Vector4f  *v4f;
    Vector3f  *v3f;
    int       *i;
  } data; 
} Uniform; 
 

extern void shader_uniform_m3x3f(Uniform *u,GLuint id, Mat3x3f *m); 
extern void shader_uniform_m4x4f(Uniform *u,GLuint id, Mat4x4f *m); 
extern void shader_uniform_v3f(Uniform *u,GLuint id, Vector3f *v); 
extern void shader_uniform_v4f(Uniform *u,GLuint id, Vector4f *v); 
extern void shader_uniform_i(Uniform *u,GLuint id, int *v); 


/* -----------------------------------------------------------------------------
   VERTEX ATTRIBUTES (a shader concept)
   -------------------------------------------------------------------------- */
// TODO: a bit misguided here, figure it out and make it work. 
#define VERTEX_INDEX   0
#define NORMAL_INDEX   1
#define COLOR_INDEX    2
#define TEXEL_INDEX    1 // TODO: cheating

typedef struct {
  GLuint vattrib;
  bool   active;
} VertexAttribute;

/* -----------------------------------------------------------------------------
   SHADER STRUCTURE
   -------------------------------------------------------------------------- */
typedef struct {
  GLuint   shader; 
  Uniform  *uniforms;
  unsigned int num_uniforms; 
  VertexAttribute attributes[4]; 
} Shader; 
#define SHADER_HAS_ATTRIB(S,X)  S->attributes[X].active
#define SHADER_ATTRIB(S,X)      S->attributes[X].vattrib





/*
   textFileRead()
   Allocates memory for a file. 
   caller is responsible for freeing that 
   memory.
*/
extern char* textFileRead(char *fn);

extern GLuint shader_load(char *fn, GLenum kind);
//extern void shader_destroy(??);

extern void shader_printCompileLog(GLuint shader);
extern void shader_printLinkLog(GLuint program);



/* returns a very basic shader */
extern Shader *shader_basic(Mat4x4f *mv, Mat4x4f *p );

#endif 
