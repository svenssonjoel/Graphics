

#include "md2.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vector.h"
#include "matrix.h"
#include "error.h"

/* Load md2 file on little-endian machine */
unsigned char *md2_loadFile(char *fn) {
  
  FILE *fp = NULL;
  unsigned long nbytes;  
  unsigned char *data = NULL;

  if ((fp = fopen(fn,"rb")) == NULL) 
    return NULL;

  
  fseek (fp , 0 , SEEK_END);
  nbytes = ftell (fp); 
  rewind (fp);
  
  printf("file is %ld bytes long\n", nbytes);
  
  data = malloc(nbytes); 

  if (data == NULL) return NULL;

  fread(data,nbytes,1,fp);
  return data;
}
/* Load md2 file on Big-endian machine.
   Will be a bit trickier */ 
unsigned char *md2_loadFileBigEndian(char *fn) {

  return NULL;
}



void md2_printHeaderInfo(unsigned char *data) {
  
  MD2Header *head; 
  head = (MD2Header*)data;
  
  printf("Magic:      %c%c%c%c\n",head->ident[0],head->ident[1],head->ident[2],head->ident[3]);
  printf("Version:    %d\n",head->version); 
  printf("Skinwidth:  %d\n", head->skinwidth);
  printf("Skinheight: %d\n", head->skinheight);
  printf("Framesize:  %d\n", head->framesize);
  printf("Num skins:  %d\n", head->num_skins);
  printf("Num verts:  %d\n", head->num_xyz);
  printf("Num tris:   %d\n", head->num_tris);
  printf("Num glcmd:  %d\n", head->num_glcmds);
  printf("Num frames: %d\n", head->num_frames);
  printf("Ofs skins:  %d\n", head->ofs_skins);
  printf("Ofs tex_st: %d\n", head->ofs_st);
  printf("Ofs tris:   %d\n", head->ofs_tris);
  printf("Ofs frames: %d\n", head->ofs_frames);
  printf("ofs glcmds: %d\n", head->ofs_glcmds);
  printf("ofs end:    %d\n", head->ofs_end);
 
}

void md2_printSkinNames(unsigned char *data){

  char str[65];
  str[64] = 0;
  
  MD2Header *head; 
  head = (MD2Header*)data;
 
  printf("Skin names:\n");
  for (int i = 0; i < head->num_skins; i++) {
    memcpy(str,data + head->ofs_skins + (i*64),64);
    printf("%s\n",str);
  }

}

void md2_printFrameiHeader(unsigned char *data, unsigned int i) {
  

  MD2Header *head; 
  MD2Frame  *frame;
  head = (MD2Header*)data;
  char str[17];  
  str[16] = 0;
  unsigned int frame_ofs;
   
  if ( i >= head->num_frames)  {
    printf("No such frame: %d\n",i);
    return;
  }
  
  frame_ofs = head->framesize * i + head->ofs_frames;
  frame = (MD2Frame *)(data + frame_ofs);

  printf("Frame: %d\n", i);
  printf("Scale: %f %f %f\n", frame->scale[0], frame->scale[1], frame->scale[2]);
  printf("Trans: %f %f %f\n", frame->translate[0], frame->translate[1], frame->translate[2]);
  
  memcpy(str,frame->name, 16); 
  printf("Name: %s \n",str);
    
}



int md2_renderFrameFlat(Shader *s, unsigned char *data, unsigned int curr_frame) {

  MD2Header *head; 
  MD2Frame  *frame;
  head = (MD2Header*)data;
  unsigned int frame_ofs;
  MD2Face *faces;
  GLuint vert;
  GLuint cols;
  GLuint ixs;

  frame_ofs = head->framesize * curr_frame + head->ofs_frames;
  frame = (MD2Frame *)&data[frame_ofs];
  faces = (MD2Face *)&data[head->ofs_tris];

  Vector3f *vertices = NULL; 
  Vector3f *colors = NULL;
  unsigned short *indices = NULL;
  vertices = (Vector3f*)malloc(head->num_xyz *  sizeof(Vector3f));
  colors = (Vector3f*)malloc(head->num_xyz *  sizeof(Vector3f));
  indices = (unsigned short*)malloc(head->num_tris *  sizeof(unsigned short) * 3);
  
  if (!vertices || !colors || !indices) {
    fprintf(stderr,"ERROR\n");
    exit(-1);
  }
  
  for ( int i = 0; i < head->num_xyz; i ++) {
    vertices[i].x = ((float)frame->verts[i].v[0]) * frame->scale[0] + frame->translate[0];
    vertices[i].y = ((float)frame->verts[i].v[1]) * frame->scale[1] + frame->translate[1];
    vertices[i].z = ((float)frame->verts[i].v[2]) * frame->scale[2] + frame->translate[2]; 
    vertices[i].x *= 0.15f;
    vertices[i].y *= 0.15f;
    vertices[i].z *= 0.15f;
   

    colors[i].x = 1.0;
    colors[i].y = 0.0;
    colors[i].z = 0.0;
  }
 for ( int i = 0; i < head->num_tris; i ++) {
    MD2Face * fs = (MD2Face*)(data+head->ofs_tris);
    indices[i*3]   = fs[i].index_xyz[0];
    indices[i*3+1] = fs[i].index_xyz[1];
    indices[i*3+2] = fs[i].index_xyz[2];
  }
  

 
  

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
  
  glGenBuffers(1, &vert);
  glBindBuffer(GL_ARRAY_BUFFER, vert);
  glBufferData(GL_ARRAY_BUFFER, head->num_xyz*sizeof(Vector3f), vertices , GL_DYNAMIC_DRAW);

  glGenBuffers(1, &cols);
  glBindBuffer(GL_ARRAY_BUFFER, cols);
  glBufferData(GL_ARRAY_BUFFER, head->num_xyz*sizeof(Vector3f), colors , GL_DYNAMIC_DRAW);

  glGenBuffers(1, &ixs);
  glBindBuffer(GL_ARRAY_BUFFER, ixs);
  glBufferData(GL_ARRAY_BUFFER, head->num_tris*3*sizeof(unsigned short) , indices , GL_DYNAMIC_DRAW);

  
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) {
    //printf("setting vertex buffer\n");
    glEnableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, vert);   
    glVertexAttribPointer(s->attributes[VERTEX_INDEX].vattrib,3,GL_FLOAT,GL_FALSE,0,(void*)0);
  }
  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX)) {
    //printf("setting color buffer\n");
    glEnableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));
    glBindBuffer(GL_ARRAY_BUFFER, cols);   
    glVertexAttribPointer(SHADER_ATTRIB(s,COLOR_INDEX),
			  3,GL_FLOAT,GL_FALSE,0,(void*)0);
  }


  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ixs);
  glDrawElements(GL_TRIANGLES,head->num_tris *3 , GL_UNSIGNED_SHORT, 0);
  //glDrawArrays(GL_TRIANGLES, 0, head->num_xyz);
  printGLError("mesh_render drawArrays:");  
  
  
  
  if (SHADER_HAS_ATTRIB(s,VERTEX_INDEX)) 
    glDisableVertexAttribArray(SHADER_ATTRIB(s,VERTEX_INDEX));
  if (SHADER_HAS_ATTRIB(s,COLOR_INDEX))
    glDisableVertexAttribArray(SHADER_ATTRIB(s,COLOR_INDEX));
 
  printGLError("mesh_render:");  
 
  glDeleteBuffers(1,&vert);
  glDeleteBuffers(1,&cols);
  glDeleteBuffers(1,&ixs);
  free(vertices);
  free(colors);
  free(indices);


  return 0;
}
