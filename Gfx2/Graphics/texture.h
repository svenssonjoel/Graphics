
#ifndef _TEXTURE_H
#define _TEXTURE_H

// #include "gl3.h"
#include "image.h"


#include <GL/glew.h>
#include <GL/gl.h>



/* old, phase out!! */
extern void initTexture2D(GLuint texID, 
			  GLint width, 
			  GLint height,
			  GLenum format,
			  GLubyte *texptr);

/* initialize a texture from a image already in memory*/
extern void texture_init2D(GLuint texID, 
			   GLint width, 
			   GLint height, 
			   GLenum format, 
			   GLubyte *texptr); 

/* same as init, but generated mipmaps */
extern void texture_init2DGenMip(GLuint texID, 
			   GLint width, 
			   GLint height, 
			   GLenum format, 
			   GLubyte *texptr); 



/* old, phase out!! */
extern void loadTexture2D(char *filename, 
		   int width, 
		   int height, 
		   int bpp, 
		   GLenum format, 
		   GLint texID);

/* initialize and load a Raw file into a texture object */
extern void texture_load2DRaw(char *filename, 
		   int width, 
		   int height, 
		   int bpp, 
		   GLenum format, 
		   GLint texID);


/* MISGUIDED! REMOVE! */
extern GLint texture_nextID();
#endif
