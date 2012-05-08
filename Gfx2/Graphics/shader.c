
#include "shader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
 
#include "error.h"
/* -----------------------------------------------------------------------------
   Uniform creation
   -------------------------------------------------------------------------- */

void shader_uniform_m3x3f(Uniform *u,GLuint id, Mat3x3f *m){
  u->type = UNIFORM_MAT3X3F;
  u->id   = id;
  u->data.m3x3f = m;
} 
void shader_uniform_m4x4f(Uniform *u,GLuint id, Mat4x4f *m){
  u->type = UNIFORM_MAT4X4F;
  u->id   = id; 
  u->data.m4x4f = m;
}

void shader_uniform_v3f(Uniform *u,GLuint id, Vector3f *v) {
  u->type = UNIFORM_VEC3F;
  u->id   = id; 
  u->data.v3f = v;
    
}
void shader_uniform_v4f(Uniform *u,GLuint id, Vector4f *v) {
  u->type = UNIFORM_VEC4F;
  u->id   = id; 
  u->data.v4f = v;
  
}

void shader_uniform_i(Uniform *u,GLuint id, int *i) {
  u->type  = UNIFORM_INT;
  u->id    = id; 
  u->data.i = i;
}

/* -----------------------------------------------------------------------------
   textFileRead
   -------------------------------------------------------------------------- */
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


/* -----------------------------------------------------------------------------
 
   -------------------------------------------------------------------------- */

//extern GLuint glCreateShader(GLuint);
//extern void glShaderSource(GLuint,GLsizei,const char**,const GLint*);

GLuint shader_load(char *fn, GLenum kind) {
  GLuint s; 
  char *prg;
  
  s = glCreateShader(kind);

  prg = textFileRead(fn); 
  if (!prg) return 0;
  // fprintf(stderr,"%s",prg);
  
  const char * prg_ = prg;
  
  glShaderSource(s,1,&prg_,NULL);
  
  free(prg);
  return s;
}
  
 
   

/* -----------------------------------------------------------------------------
   printCompileLog(GLuint shader) 
   -------------------------------------------------------------------------- */
void shader_printCompileLog(GLuint shader) {
  char *log;
  int   length;
  
  glGetShaderiv(shader,GL_INFO_LOG_LENGTH, &length);
  log = (char*)malloc(length);
  glGetShaderInfoLog(shader,length,NULL,log);
  fprintf(stdout,"%s\n",log);
  free(log);

}

/* -----------------------------------------------------------------------------
   void printLinkLog(GLuint program);
   -------------------------------------------------------------------------- */
void shader_printLinkLog(GLuint prg) {
  char *log;
  int   length;
  
  glGetProgramiv(prg,GL_INFO_LOG_LENGTH, &length);
  log = (char*)malloc(length);
  glGetProgramInfoLog(prg,length,NULL,log);
  fprintf(stdout,"%s\n",log);
  free(log);

}
/* -----------------------------------------------------------------------------
   void printLinkLog(GLuint program);
   -------------------------------------------------------------------------- */

Shader *shader_basic(Mat4x4f *mv,Mat4x4f *p ) {
 
  
  Shader *s = NULL;
 

  GLuint shader_program;
  GLuint shader_frag;
  GLuint shader_vert; 

  GLuint shader_vindex;
  GLuint shader_cindex;

  GLuint shader_proj;
  GLuint shader_mod;


  char vertexshader[] = 
    "#version 140\n" 
    "uniform mat4 proj;\n"
    "uniform mat4 mod;\n"
    "in vec4 Vertex;\n"
    "in vec4 Color;\n"
    "out vec4 ColorInterp;\n"
    "void main() {\n"
    "gl_Position = proj * (mod * Vertex);\n"
    "ColorInterp = Color;\n}";

  char fragmentshader[] = 
    "#version 140\n"
    "in vec4 ColorInterp;\n"
    "out vec4 out_FragColor;\n"
    "void main() {\n"
    "out_FragColor = ColorInterp;\n}";
  

  const char * f = fragmentshader;
  const char * v = vertexshader;
  //printf("%s\n",vertexshader);
  //printf("\n\n%s\n",fragmentshader);
  


  
  s = (Shader*)malloc(sizeof(Shader));
  
  if (s == NULL) 
    return s;
  

  shader_vert = glCreateShader(GL_VERTEX_SHADER);
  shader_frag = glCreateShader(GL_FRAGMENT_SHADER);
  
  //shader_vert = shader_load("/home/ian/Graphics/Graphics/Gfx2/Shaders/const_mv.vert",GL_VERTEX_SHADER);
  //shader_frag = shader_load("/home/ian/Graphics/Graphics/Gfx2/Shaders/const_mv.frag",GL_FRAGMENT_SHADER);
  
  printGLError("shader A:");
  
  if (!shader_vert) printf("error creating shader\n");
  if (!shader_frag) printf("error creating shader\n");

  
  glShaderSource(shader_vert,1,&v,NULL);
  glShaderSource(shader_frag,1,&f,NULL);
  
  printGLError("shader B:");
  
  glCompileShader(shader_vert);
  shader_printCompileLog(shader_vert);
  glCompileShader(shader_frag);
  shader_printCompileLog(shader_frag);
 
  printGLError("shader C:");   

  shader_program = glCreateProgram();
  		
  glAttachShader(shader_program,shader_vert);
  glAttachShader(shader_program,shader_frag);
  
  printGLError("shader D:");
  
  glLinkProgram(shader_program);
  
  printGLError("shader E:");

  shader_proj = glGetUniformLocation(shader_program, "proj");
   printGLError("shader F1:");
  shader_mod  = glGetUniformLocation(shader_program, "mod");
   printGLError("shader F2:");
 
  
  shader_vindex = glGetAttribLocation(shader_program, "Vertex");
  shader_cindex = glGetAttribLocation(shader_program, "Color");
  printGLError("shader G:");
  
  //  shader_uniform_m4x4f(&unis[0],shader_proj,p);
  //shader_uniform_m4x4f(&unis[1],shader_mod,mv);

  s->uniforms = (Uniform*)malloc(2*sizeof(Uniform));
  if (s->uniforms == NULL) printf("ERROR\n");
  s->num_uniforms = 2; 
  
  shader_uniform_m4x4f(&s->uniforms[0],shader_proj,p);
  shader_uniform_m4x4f(&s->uniforms[1],shader_mod,mv);

  
  s->shader   = shader_program;
  s->attributes[VERTEX_INDEX].vattrib = shader_vindex;
  s->attributes[VERTEX_INDEX].active = true;
  
  s->attributes[COLOR_INDEX].vattrib = shader_cindex;
  s->attributes[COLOR_INDEX].active = true;
  
  return s;
  
}
 
