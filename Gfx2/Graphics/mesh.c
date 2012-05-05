
#include "mesh.h"


#include "error.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <stdbool.h>

/* 3ds file loading */ 
#include <lib3ds/file.h>
#include <lib3ds/mesh.h>

/* -----------------------------------------------------------------------------
   Set all fields to 0
   -------------------------------------------------------------------------- */
Mesh * mesh_init(Mesh *m) {
  if (!m) return NULL; 
  
  m->vertices = NULL;
  m->num_vertices = 0;
  m->normals  = NULL; 
  m->colors   = NULL; 
  m->texels = NULL;
  
  m->vao = 0; 
  m->vertex_buffer = 0;
  m->normal_buffer = 0;
  m->color_buffer = 0; 
  m->texel_buffer = 0;
  
  // m->texture_id = 0;
  // m->textured  = false;
  
  m->indices = NULL;
  m->indices_type = 0;
  
  return m;
}

/* -----------------------------------------------------------------------------
   Create a mesh. 
   -------------------------------------------------------------------------- */
Mesh *mesh_create(){
  
  Mesh *m = NULL; 

  m = (Mesh *)malloc(sizeof(Mesh)); 

  return (mesh_init(m));
}



/* -----------------------------------------------------------------------------
   Destroy a mesh and all associated data
   -------------------------------------------------------------------------- */
void mesh_destroy(Mesh *m) {
  if (m) 
    free(m);
}

/* -----------------------------------------------------------------------------
   SetShaders
   -------------------------------------------------------------------------- */
//void mesh_setShader(Mesh *m, GLuint sh, Uniform *uniforms){
  //m->shader = sh; 
  //m->uniforms = uniforms;
  
//}

/* -----------------------------------------------------------------------------
   Upload a mesh (that is send the data to the GPU) 
   -------------------------------------------------------------------------- */
void mesh_upload(Mesh *m) { 

  if (m->vertices) {
    glGenBuffers(1, &m->vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m->vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, 
		 m->num_vertices*sizeof(Vector3f),
		 m->vertices, 
		 GL_STATIC_DRAW);
  }
  if (m->normals) { 
    glGenBuffers(1, &m->normal_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m->normal_buffer);
    glBufferData(GL_ARRAY_BUFFER, 
		 m->num_vertices*sizeof(Vector3f),
		 m->normals, 
		 GL_STATIC_DRAW);
  } 
  if (m->colors) { 
    glGenBuffers(1, &m->color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m->color_buffer);
    glBufferData(GL_ARRAY_BUFFER, 
		 m->num_vertices*sizeof(Vector3f),
		 m->colors, 
		 GL_STATIC_DRAW);
  } 
  if (m->texels) { 
    glGenBuffers(1, &m->texel_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m->texel_buffer);
    glBufferData(GL_ARRAY_BUFFER, 
		 m->num_vertices*sizeof(Vector2f),
		 m->texels, 
		 GL_STATIC_DRAW); 
  } 
  if (m->indices) { 
    glGenBuffers(1, &m->index_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m->index_buffer);
    if (m->indices_type == GL_UNSIGNED_BYTE) {
      glBufferData(GL_ARRAY_BUFFER, 
		   m->num_indices*sizeof(GLubyte),
		   m->indices, 
		   GL_STATIC_DRAW);
    }
    else if (m->indices_type == GL_UNSIGNED_SHORT) {
      glBufferData(GL_ARRAY_BUFFER, 
		   m->num_indices*sizeof(GLushort),
		   m->indices, 
		   GL_STATIC_DRAW);
    }
    else if (m->indices_type == GL_UNSIGNED_INT) {
      glBufferData(GL_ARRAY_BUFFER, 
		   m->num_indices*sizeof(GLuint),
		   m->indices, 
		   GL_STATIC_DRAW);
    }
  } 
 
}

// TODO: Fix this, the vertex attribute indices needs 
//       To be managed in some other way. 
//       It is up to the programmer to use a matching shader.. 
//       That is if you use a Mesh with Vertices, Normals and Texels 
//       you must use a shader that takes the vertices at attrib 0 
//                             that takes the normals at attrib 1 
//                             that takes the texels at attrib 2 
void mesh_upload_prim(Mesh *m) { 
  
  glGenVertexArrays(1,&m->vao);
  glBindVertexArray(m->vao);
  printGLError("u1");
  if (m->vertices) {
    glGenBuffers(1, &m->vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m->vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, 
		 m->num_vertices*sizeof(Vector3f),
		 m->vertices, 
		 GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(VERTEX_INDEX);
    glVertexAttribPointer(VERTEX_INDEX,3,GL_FLOAT,GL_FALSE,0,(const GLvoid *)0);

    printGLError("u2");
  }
  if (m->normals) { 
    glGenBuffers(1, &m->normal_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m->normal_buffer);
    glBufferData(GL_ARRAY_BUFFER, 
		 m->num_vertices*sizeof(Vector3f),
		 m->normals, 
		 GL_STATIC_DRAW);
    glEnableVertexAttribArray(NORMAL_INDEX);
    glVertexAttribPointer(NORMAL_INDEX,3,GL_FLOAT,GL_FALSE,0,(const GLvoid *)0);

    printGLError("u3");
  } 
  if (m->colors) { 
    glGenBuffers(1, &m->color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m->color_buffer);
    glBufferData(GL_ARRAY_BUFFER, 
		 m->num_vertices*sizeof(Vector3f),
		 m->colors, 
		 GL_STATIC_DRAW);
    glEnableVertexAttribArray(COLOR_INDEX);
    glVertexAttribPointer(COLOR_INDEX,3,GL_FLOAT,GL_FALSE,0,(const GLvoid *)0);

    printGLError("u4");
    } 
  if (m->texels) { 
    glGenBuffers(1, &m->texel_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m->texel_buffer);
    glBufferData(GL_ARRAY_BUFFER, 
		 m->num_vertices*sizeof(Vector2f),
		 m->texels, 
		 GL_STATIC_DRAW);

    glEnableVertexAttribArray(TEXEL_INDEX);
    glVertexAttribPointer(TEXEL_INDEX,2,GL_FLOAT,GL_FALSE,0,(const GLvoid *)0); 
    
    printGLError("u5");
  } 
  if (m->indices) { 
    glGenBuffers(1, &m->index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->index_buffer);
    if (m->indices_type == GL_UNSIGNED_BYTE) {
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
		   m->num_indices*sizeof(GLubyte),
		   m->indices, 
		   GL_STATIC_DRAW);
    }
    else if (m->indices_type == GL_UNSIGNED_SHORT) {
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
		   m->num_indices*sizeof(GLushort),
		   m->indices, 
		   GL_STATIC_DRAW);
    }
    else if (m->indices_type == GL_UNSIGNED_INT) {
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
		   m->num_indices*sizeof(GLuint),
		   m->indices, 
		   GL_STATIC_DRAW);
    }
  } 
  printGLError("u6");
  
  //glBindVertexArray(0);
}


/* -----------------------------------------------------------------------------
   renderDot. Render the vertices as dots
   -------------------------------------------------------------------------- */
void mesh_renderDot(Shader *s,Mesh *m){
  glUseProgram(s->shader);
  
  //printf("%d\n",m->shader);
  printGLError("mesh_render:");

  /* set uniforms */
  for (int i = 0; i < s->num_uniforms; ++i) {
    switch (s->uniforms[i].type) {
    case UNIFORM_UNDEFINED:
      fprintf(stderr,"mesh_render: Undefined uniform\n");
      exit(EXIT_FAILURE);
    case UNIFORM_MAT3X3F:
      glUniformMatrix3fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m3x3f);
      break;
    case UNIFORM_MAT4X4F:
      glUniformMatrix4fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m4x4f);
      break;
    case UNIFORM_VEC3F:
      glUniform3f(s->uniforms[i].id,
		  s->uniforms[i].data.v3f->x,
		  s->uniforms[i].data.v3f->y,
		  s->uniforms[i].data.v3f->z);
      break;
    case UNIFORM_VEC4F:
      glUniform4f(s->uniforms[i].id,
      		  s->uniforms[i].data.v4f->x,
		  s->uniforms[i].data.v4f->y,
		  s->uniforms[i].data.v4f->z,
		  s->uniforms[i].data.v4f->w);
      break;
    case UNIFORM_INT:
      glUniform1i(s->uniforms[i].id,
		  *s->uniforms[i].data.i);
      break;
      
    default: 
      fprintf(stderr,"mesh_render: Uniform fall through!\n");
      exit(EXIT_FAILURE);

    }
    
  } // uniforms should be set! 

  /* Experiment. try with vertex buffer only and render dots */
  //  glEnableVertexAttribArray(m->vindex);
  glEnableVertexAttribArray(s->attributes[VERTEX_INDEX].vattrib);
  printGLError("mesh_render A:");  

  glBindBuffer(GL_ARRAY_BUFFER, m->vertex_buffer);
  printGLError("mesh_render B:");  
  glVertexAttribPointer(s->attributes[VERTEX_INDEX].vattrib,
			3, 
			GL_FLOAT, 
			GL_FALSE,
			0,
			(void*)0);
  printGLError("mesh_render C:");  
  glDrawArrays(GL_POINTS,0,m->num_vertices);
  printGLError("mesh_render D:");  
  glDisableVertexAttribArray(s->attributes[VERTEX_INDEX].vattrib);

  printGLError("mesh_render:");
  

}

/* -----------------------------------------------------------------------------
   renderFill. fill with flat color
   -------------------------------------------------------------------------- */
void mesh_renderFill(Shader *s,Mesh *m){
  glUseProgram(s->shader);
  
  //printf("%d\n",m->shader);
  printGLError("mesh_render:");

  /* set uniforms */
  for (int i = 0; i < s->num_uniforms; ++i) {
    switch (s->uniforms[i].type) {
    case UNIFORM_UNDEFINED:
      fprintf(stderr,"mesh_render: Undefined uniform\n");
      exit(EXIT_FAILURE);
    case UNIFORM_MAT3X3F:
      glUniformMatrix3fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m3x3f);
      break;
    case UNIFORM_MAT4X4F:
      glUniformMatrix4fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m4x4f);
      break;
    case UNIFORM_VEC3F:
      glUniform3f(s->uniforms[i].id,
		  s->uniforms[i].data.v3f->x,
		  s->uniforms[i].data.v3f->y,
		  s->uniforms[i].data.v3f->z);
      break;
    case UNIFORM_VEC4F:
      glUniform4f(s->uniforms[i].id,
      		  s->uniforms[i].data.v4f->x,
		  s->uniforms[i].data.v4f->y,
		  s->uniforms[i].data.v4f->z,
		  s->uniforms[i].data.v4f->w);
      break;
    case UNIFORM_INT:
      glUniform1i(s->uniforms[i].id,
		  *s->uniforms[i].data.i);
      break;
  
    default: 
      fprintf(stderr,"mesh_render: Uniform fall through!\n");
      //exit(EXIT_FAILURE);

    }
    
  } // uniforms should be set! 

  glEnableVertexAttribArray(s->attributes[VERTEX_INDEX].vattrib);
  printGLError("mesh_render A:");  

  glBindBuffer(GL_ARRAY_BUFFER, m->vertex_buffer);
  printGLError("mesh_render B:");  
  glVertexAttribPointer(s->attributes[VERTEX_INDEX].vattrib,
			3, GL_FLOAT, GL_FALSE,0,(void*)0);
  printGLError("mesh_render C:");  
  
  // bind the index buffer (points out the triangles to fill) 
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->index_buffer);

  //glDrawArrays(GL_POINTS,0,m->num_vertices);
  glDrawElements(GL_TRIANGLES,m->num_indices,m->indices_type,0);

  printGLError("mesh_render D:");  
  glDisableVertexAttribArray(s->attributes[VERTEX_INDEX].vattrib);

  printGLError("mesh_render:");
  

}

void mesh_renderFillLit(Shader *s,Mesh *m){
  glUseProgram(s->shader);  
 
  /* set uniforms */
  for (int i = 0; i < s->num_uniforms; ++i) {
    switch (s->uniforms[i].type) {
    case UNIFORM_UNDEFINED:
      fprintf(stderr,"mesh_render: Undefined uniform\n");
      exit(EXIT_FAILURE);
    case UNIFORM_MAT3X3F:
      glUniformMatrix3fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m3x3f);
      printGLError("mesh_render (MAT3X3): ");  
      break;
    case UNIFORM_MAT4X4F:
      glUniformMatrix4fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m4x4f);
      printGLError("mesh_render (MAT4X4):");  
      break;
    case UNIFORM_VEC3F:
      glUniform3f(s->uniforms[i].id,
		  s->uniforms[i].data.v3f->x,
		  s->uniforms[i].data.v3f->y,
		  s->uniforms[i].data.v3f->z);
      printGLError("mesh_render (VEC3):");  
      break;
    case UNIFORM_VEC4F:
      glUniform4f(s->uniforms[i].id,
      		  s->uniforms[i].data.v4f->x,
		  s->uniforms[i].data.v4f->y,
		  s->uniforms[i].data.v4f->z,
		  s->uniforms[i].data.v4f->w);
      printGLError("mesh_render (VEC4):");  
      break;
    case UNIFORM_INT:
      glUniform1i(s->uniforms[i].id,
		  *s->uniforms[i].data.i);
      printGLError("mesh_render (INT):");  
      break;
    default: 
      /* Silently fall through */
      break;
    }
    
  } // uniforms should be set! 
  printGLError("mesh_render (UNIFORMS):");  

  /* VERTICES */
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, m->vertex_buffer);
    glVertexAttribPointer(SHADER_ATTRIB(s,VERTEX_INDEX),
			  3, GL_FLOAT, GL_FALSE,0,(void*)0);
  }
  
  /* TEXTURE COORDS */   
  if (SHADER_HAS_ATTRIB(s,TEXEL_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,TEXEL_INDEX));
    
    glBindBuffer(GL_ARRAY_BUFFER, m->texel_buffer);   
    glVertexAttribPointer(SHADER_ATTRIB(s,TEXEL_INDEX),
			  2,GL_FLOAT,GL_FALSE,0,(void*)0);
  }
  
  /* NORMALS */
  if (SHADER_HAS_ATTRIB(s,NORMAL_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,NORMAL_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, m->normal_buffer);   
    glVertexAttribPointer(SHADER_ATTRIB(s,NORMAL_INDEX),
			  3,GL_FLOAT,GL_FALSE,0,(void*)0);
  }

  /* COLORS */
  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, m->color_buffer);   
    glVertexAttribPointer(SHADER_ATTRIB(s,COLOR_INDEX),
			  3,GL_FLOAT,GL_FALSE,0,(void*)0);
  }


  // bind the index buffer (points out the triangles to fill) 
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->index_buffer);

  glDrawElements(GL_TRIANGLES,m->num_indices,m->indices_type,0);

  
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));
  if (SHADER_HAS_ATTRIB(s,NORMAL_INDEX))
    glDisableVertexAttribArray(SHADER_ATTRIB(s,NORMAL_INDEX));
  if (SHADER_HAS_ATTRIB(s,TEXEL_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,TEXEL_INDEX));
  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));
  printGLError("mesh_render:");  
 
}

void mesh_render(Shader *s, Mesh *m){
  glUseProgram(s->shader);

  printGLError("mesh_render (USEPROGRAM):");
 
  /* set uniforms */
  for (int i = 0; i < s->num_uniforms; ++i) {
    switch (s->uniforms[i].type) {
    case UNIFORM_UNDEFINED:
      fprintf(stderr,"mesh_render: Undefined uniform\n");
      exit(EXIT_FAILURE);
    case UNIFORM_MAT3X3F:
      glUniformMatrix3fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m3x3f);
      printGLError("mesh_render (MAT3X3): ");  
      break;
    case UNIFORM_MAT4X4F:
      glUniformMatrix4fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m4x4f);
      printGLError("mesh_render (MAT4X4):");  
      break;
    case UNIFORM_VEC3F:
      glUniform3f(s->uniforms[i].id,
		  s->uniforms[i].data.v3f->x,
		  s->uniforms[i].data.v3f->y,
		  s->uniforms[i].data.v3f->z);
      printGLError("mesh_render (VEC3):");  
      break;
    case UNIFORM_VEC4F:
      glUniform4f(s->uniforms[i].id,
      		  s->uniforms[i].data.v4f->x,
		  s->uniforms[i].data.v4f->y,
		  s->uniforms[i].data.v4f->z,
		  s->uniforms[i].data.v4f->w);
      printGLError("mesh_render (VEC4):");  
      break;
    case UNIFORM_INT:
      glUniform1i(s->uniforms[i].id,
		  *s->uniforms[i].data.i);
      printGLError("mesh_render (INT):");  
      break;
    default: 
      /* Silently fall through */
      break;
    }
    
  } // uniforms should be set! 
  printGLError("mesh_render (UNIFORMS):");  

  /* VERTICES */
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) {
    printGLError("mesh_render (VERTICES0):");  
    glEnableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));
    printGLError("mesh_render (VERTICES1):");  
    glBindBuffer(GL_ARRAY_BUFFER, m->vertex_buffer);
    printGLError("mesh_render (VERTICES2):");  
    glVertexAttribPointer(SHADER_ATTRIB(s,VERTEX_INDEX),
			  3, GL_FLOAT, GL_FALSE,0,(void*)0);
    printGLError("mesh_render (VERTICES3):");  
  }
  printGLError("mesh_render (VERTICES):");  
  
  /* TEXTURE COORDS */   
  if (SHADER_HAS_ATTRIB(s,TEXEL_INDEX)) {
    
    glEnableVertexAttribArray(SHADER_ATTRIB(s,TEXEL_INDEX));
    
    glBindBuffer(GL_ARRAY_BUFFER, m->texel_buffer);   
    glVertexAttribPointer(SHADER_ATTRIB(s,TEXEL_INDEX),
			  2,GL_FLOAT,GL_FALSE,0,(void*)0);
  }
  printGLError("mesh_render (TEXCOORDS):");  
  
  /* NORMALS */
  if (SHADER_HAS_ATTRIB(s,NORMAL_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,NORMAL_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, m->normal_buffer);   
    glVertexAttribPointer(SHADER_ATTRIB(s,NORMAL_INDEX),
			  3,GL_FLOAT,GL_FALSE,0,(void*)0);
  }
  printGLError("mesh_render (NORMALS):");  

  /* COLORS */
  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, m->color_buffer);   
    glVertexAttribPointer(SHADER_ATTRIB(s,COLOR_INDEX),
			  3,GL_FLOAT,GL_FALSE,0,(void*)0);
  }
  printGLError("mesh_render (COLOR):");  

  // bind the index buffer (points out the triangles to fill) 
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->index_buffer);
  printGLError("mesh_render (INDICES):");  

  glDrawElements(GL_TRIANGLES,m->num_indices,m->indices_type,0);
  printGLError("mesh_render (DrawElements):");  
  
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));
  if (SHADER_HAS_ATTRIB(s,NORMAL_INDEX))
    glDisableVertexAttribArray(SHADER_ATTRIB(s,NORMAL_INDEX));
  if (SHADER_HAS_ATTRIB(s,TEXEL_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,TEXEL_INDEX));
  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));
  printGLError("mesh_render END:");  
 
}


void mesh_renderTex_prim(Shader *s,GLuint textureID, Mesh *m){
  printGLError("mesh_render (BEFORE USEPROGRAM):");
  glUseProgram(s->shader);

  printGLError("mesh_render (USEPROGRAM):");
  /* ACTIVATE TEXTURING */
  
  glActiveTexture(GL_TEXTURE0);
  
  printGLError("mesh_render (TEXTURE1):");
  glBindTexture(GL_TEXTURE_2D, textureID);

  printGLError("mesh_render (TEXTURE2):");
  

  /* set uniforms */
  for (int i = 0; i < s->num_uniforms; ++i) {
    switch (s->uniforms[i].type) {
    case UNIFORM_UNDEFINED:
      fprintf(stderr,"mesh_render: Undefined uniform\n");
      exit(EXIT_FAILURE);
    case UNIFORM_MAT3X3F:
      glUniformMatrix3fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m3x3f);
      printGLError("mesh_render (MAT3X3): ");  
      break;
    case UNIFORM_MAT4X4F:
      glUniformMatrix4fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m4x4f);
      printGLError("mesh_render (MAT4X4):");  
      break;
    case UNIFORM_VEC3F:
      glUniform3f(s->uniforms[i].id,
		  s->uniforms[i].data.v3f->x,
		  s->uniforms[i].data.v3f->y,
		  s->uniforms[i].data.v3f->z);
      printGLError("mesh_render (VEC3):");  
      break;
    case UNIFORM_VEC4F:
      glUniform4f(s->uniforms[i].id,
      		  s->uniforms[i].data.v4f->x,
		  s->uniforms[i].data.v4f->y,
		  s->uniforms[i].data.v4f->z,
		  s->uniforms[i].data.v4f->w);
      printGLError("mesh_render (VEC4):");  
      break;
    case UNIFORM_INT:
      glUniform1i(s->uniforms[i].id,
		  *s->uniforms[i].data.i);
      printGLError("mesh_render (INT):");  
      break;
    default: 
      /* Silently fall through */
      break;
    }
    
  } // uniforms should be set! 
  printGLError("mesh_render (UNIFORMS):");  

  
  glBindVertexArray(m->vao);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->index_buffer);
  glDrawElements(GL_TRIANGLES,m->num_indices,m->indices_type,0);

  
  
 
}

void mesh_renderTex(Shader *s,GLuint textureID, Mesh *m){
  glUseProgram(s->shader);

  printGLError("mesh_render (USEPROGRAM):");
  /* ACTIVATE TEXTURING */
  
  glActiveTexture(GL_TEXTURE0);
  
  printGLError("mesh_render (TEXTURE1):");
  glBindTexture(GL_TEXTURE_2D, textureID);

  printGLError("mesh_render (TEXTURE2):");
  

 
  /* set uniforms */
  for (int i = 0; i < s->num_uniforms; ++i) {
    switch (s->uniforms[i].type) {
    case UNIFORM_UNDEFINED:
      fprintf(stderr,"mesh_render: Undefined uniform\n");
      exit(EXIT_FAILURE);
    case UNIFORM_MAT3X3F:
      glUniformMatrix3fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m3x3f);
      printGLError("mesh_render (MAT3X3): ");  
      break;
    case UNIFORM_MAT4X4F:
      glUniformMatrix4fv(s->uniforms[i].id, 1, GL_FALSE, *s->uniforms[i].data.m4x4f);
      printGLError("mesh_render (MAT4X4):");  
      break;
    case UNIFORM_VEC3F:
      glUniform3f(s->uniforms[i].id,
		  s->uniforms[i].data.v3f->x,
		  s->uniforms[i].data.v3f->y,
		  s->uniforms[i].data.v3f->z);
      printGLError("mesh_render (VEC3):");  
      break;
    case UNIFORM_VEC4F:
      glUniform4f(s->uniforms[i].id,
      		  s->uniforms[i].data.v4f->x,
		  s->uniforms[i].data.v4f->y,
		  s->uniforms[i].data.v4f->z,
		  s->uniforms[i].data.v4f->w);
      printGLError("mesh_render (VEC4):");  
      break;
    case UNIFORM_INT:
      glUniform1i(s->uniforms[i].id,
		  *s->uniforms[i].data.i);
      printGLError("mesh_render (INT):");  
      break;
    default: 
      /* Silently fall through */
      break;
    }
    
  } // uniforms should be set! 
  printGLError("mesh_render (UNIFORMS):");  

  /* VERTICES */
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) {
    printGLError("mesh_render (VERTICES0):");  
    glEnableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));
    printGLError("mesh_render (VERTICES1):");  
    glBindBuffer(GL_ARRAY_BUFFER, m->vertex_buffer);
    printGLError("mesh_render (VERTICES2):");  
    
    glVertexAttribPointer(SHADER_ATTRIB(s,VERTEX_INDEX),
			  3, GL_FLOAT, GL_FALSE,0,(void*)0);
    printGLError("mesh_render (VERTICES3):");  
  }
  printGLError("mesh_render (VERTICES):");  
  
  /* TEXTURE COORDS */   
  if (SHADER_HAS_ATTRIB(s,TEXEL_INDEX)) {
    
    glEnableVertexAttribArray(SHADER_ATTRIB(s,TEXEL_INDEX));
    
    glBindBuffer(GL_ARRAY_BUFFER, m->texel_buffer);   
    glVertexAttribPointer(SHADER_ATTRIB(s,TEXEL_INDEX),
			  2,GL_FLOAT,GL_FALSE,0,(void*)0);
  }
  printGLError("mesh_render (TEXCOORDS):");  
  
  /* NORMALS */
  if (SHADER_HAS_ATTRIB(s,NORMAL_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,NORMAL_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, m->normal_buffer);   
    glVertexAttribPointer(SHADER_ATTRIB(s,NORMAL_INDEX),
			  3,GL_FLOAT,GL_FALSE,0,(void*)0);
  }
  printGLError("mesh_render (NORMALS):");  

  /* COLORS */
  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX)) {
    glEnableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, m->color_buffer);   
    glVertexAttribPointer(SHADER_ATTRIB(s,COLOR_INDEX),
			  3,GL_FLOAT,GL_FALSE,0,(void*)0);
  }
  printGLError("mesh_render (COLOR):");  

  // bind the index buffer (points out the triangles to fill) 
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->index_buffer);
  printGLError("mesh_render (INDICES):");  

  glDrawElements(GL_TRIANGLES,m->num_indices,m->indices_type,0);
  printGLError("mesh_render (DrawElements):");  
  
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));
  if (SHADER_HAS_ATTRIB(s,NORMAL_INDEX))
    glDisableVertexAttribArray(SHADER_ATTRIB(s,NORMAL_INDEX));
  if (SHADER_HAS_ATTRIB(s,TEXEL_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,TEXEL_INDEX));
  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));
  printGLError("mesh_render END:");  
 
}



/* -----------------------------------------------------------------------------
   Simple 3ds loader. 
   uses lib3ds, 
   loads only "mesh 0" from the file    
   -------------------------------------------------------------------------- */
/* TODO: fix return value */
int mesh_load3ds(Mesh *m, char* filename, unsigned int mode) {
  
  Lib3dsFile *model = NULL;
  //int n_meshes = 0;
  int n_faces  = 0;
  Lib3dsMesh *mesh;
  int n_points = 0;
  //int n_texels = 0;

  model = lib3ds_file_load(filename);
  mesh  = model->meshes; 
  
  
  n_faces = mesh->faces;

  n_points = n_faces * 3;

  m->vertices = (Vector3f *)malloc(n_points*sizeof(Vector3f));
  if (mode | MESH_LOAD_TEXELS) 
    m->texels = (Vector2f *)malloc(n_points*sizeof(Vector2f));  
  if (mode | MESH_LOAD_NORMALS) 
    m->normals = (Vector3f *)malloc(n_points*sizeof(Vector3f));

  m->num_vertices = n_points;
  m->indices = (unsigned char*)malloc(n_points*sizeof(unsigned int));
  m->num_indices = n_points;
  m->indices_type = GL_UNSIGNED_INT;

  /* extract all the vertices */
  int index = 0; 
  for (int i = 0; i < n_faces; ++i) {
    m->vertices[index].x   = mesh->pointL[mesh->faceL[i].points[0]].pos[0]; 
    m->vertices[index].y   = mesh->pointL[mesh->faceL[i].points[0]].pos[1]; 
    m->vertices[index++].z = mesh->pointL[mesh->faceL[i].points[0]].pos[2]; 
    m->vertices[index].x   = mesh->pointL[mesh->faceL[i].points[1]].pos[0]; 
    m->vertices[index].y   = mesh->pointL[mesh->faceL[i].points[1]].pos[1]; 
    m->vertices[index++].z = mesh->pointL[mesh->faceL[i].points[1]].pos[2]; 
    m->vertices[index].x   = mesh->pointL[mesh->faceL[i].points[2]].pos[0]; 
    m->vertices[index].y   = mesh->pointL[mesh->faceL[i].points[2]].pos[1]; 
    m->vertices[index++].z = mesh->pointL[mesh->faceL[i].points[2]].pos[2]; 
  }

    // memcpy(m->vertices,mesh->pointL,n_points*sizeof(Vector3f));
  
  unsigned int * inds = (unsigned int *)m->indices;
  index = 0; 
  for (int i = 0; i < n_faces; i ++) {
    for (int j = 0; j < 3; j ++) {
      
      inds[index] = index; // (unsigned int)mesh->faceL[i].points[j];
      printf("%d", inds[index]);
	     index ++;
    }
  }

  Lib3dsVector *n = malloc(n_points * sizeof(Lib3dsVector));
  

  lib3ds_mesh_calculate_normals(mesh,n);
  
  //memcpy(m->normals,n,n_points*sizeof(Vector3f));
  for (int i = 0; i < n_points; i ++) {
    m->normals[i].x = n[i][0];
    m->normals[i].y = n[i][1];
    m->normals[i].z = n[i][2];
    printf("%f %f %f \n",
	   m->normals[i].x,
	   m->normals[i].y,
	   m->normals[i].z);
  }
  
  return 0;
}



/* There are a number of loose ends here 
   
   
 */
void mesh_allocateAttribs(Mesh *m, unsigned int n, unsigned long mode) {
  
  if (mode & MESH_VERTICES) 
    m->vertices = (Vector3f *)malloc(n * sizeof(Vector3f)); 
  if (mode & MESH_TEXELS) 
    m->texels = (Vector2f *)malloc(n * sizeof(Vector2f));
  if (mode & MESH_COLORS) 
    m->colors = (Vector3f *)malloc(n * sizeof(Vector3f));
  if (mode & MESH_NORMALS) 
    m->normals = (Vector3f *)malloc(n * sizeof(Vector3f));

  m->num_vertices = n;
}

void mesh_freeAttribs(Mesh *m) {
  if (m->vertices) 
    free(m->vertices);
  if (m->texels) 
    free(m->texels);
  if (m->colors)
    free(m->colors);
  if (m->normals)
    free(m->normals);
}
