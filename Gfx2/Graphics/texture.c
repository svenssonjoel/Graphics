/* texture.c */

#include "texture.h"


//#include <GL/gl.h>
//#include <GL/freeglut.h>  



#include "error.h"
#include "image.h"

GLint texture_currentId = 12; 

void initTexture2D(GLuint texID, 
		   GLint width, 
		   GLint height,
		   GLenum format,
		   GLubyte *texptr){

   
  glBindTexture(GL_TEXTURE_2D, texID);
  printGLError("binding texture:");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
 
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,0, format,
	       GL_UNSIGNED_BYTE, texptr);
  printGLError("teximage2d:");

}

void texture_init2D(GLuint texID, 
		   GLint width, 
		   GLint height,
		   GLenum format,
		   GLubyte *texptr){

   
  glBindTexture(GL_TEXTURE_2D, texID);
  printGLError("binding texture:");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
 
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,0, format,
	       GL_UNSIGNED_BYTE, texptr);
  printGLError("teximage2d:");

}

void texture_init2DGenMip(GLuint texID, 
		   GLint width, 
		   GLint height,
		   GLenum format,
		   GLubyte *texptr){

   
  glBindTexture(GL_TEXTURE_2D, texID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
 
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,0, format,
	       GL_UNSIGNED_BYTE, texptr);
  
  glGenerateMipmap(GL_TEXTURE_2D);
  
  printGLError("teximage2d:");

}


void loadTexture2D(char *filename, int width, int height, int bpp, GLenum format, GLint texID) {
  
  Image *img;
  
  img = image_create(width,height,bpp);
  
  image_loadRaw(filename,img);

  initTexture2D(texID,width,height,format,img->data);
  
  image_destroy(img);

}

void texture_load2DRaw(char *filename, int width, int height, int bpp, GLenum format, GLint texID) {
  
  Image *img;
  
  img = image_create(width,height,bpp);
  
  image_loadRaw(filename,img);

  initTexture2D(texID,width,height,format,img->data);
  
  image_destroy(img);

}


/* this already exists, glGenTextures */
GLint texture_nextID() {
  return (texture_currentId ++);
}
