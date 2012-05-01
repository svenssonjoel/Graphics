#ifndef __MD2_H
#define __MD2_H


#include "shader.h"
#include <stdint.h>

typedef struct
{
  uint8_t    ident[4];              // magic number. must be equal to "IDP2"
  int32_t     version;            // md2 version. must be equal to 8
  
  int32_t     skinwidth;          // width of the texture
  int32_t     skinheight;         // height of the texture
  int32_t     framesize;          // size of one frame in bytes
  
  int32_t     num_skins;          // number of textures
  int32_t     num_xyz;            // number of vertices
  int32_t     num_st;             // number of texture coordinates
  int32_t     num_tris;           // number of triangles
  int32_t     num_glcmds;         // number of opengl commands
  int32_t     num_frames;         // total number of frames
  
  int32_t     ofs_skins;          // offset to skin names (64 bytes each)
  int32_t     ofs_st;             // offset to s-t texture coordinates
  int32_t     ofs_tris;           // offset to triangles
  int32_t     ofs_frames;         // offset to frame data
  int32_t     ofs_glcmds;         // offset to opengl commands
  int32_t     ofs_end;            // offset to end of file

} MD2Header;

typedef struct
{
  uint8_t  v[3];                // compressed vertex (x, y, z) coordinates
  uint8_t  lightnormalindex;    // index to a normal vector for the lighting

} MD2Vertex;

typedef struct
{
  uint16_t   s;
  uint16_t   t;

} MD2Texel;

typedef struct {
  float     scale[3];
  float     translate[3];
  uint8_t   name[16];
  MD2Vertex verts[1];
  
} MD2Frame;

typedef struct
{
    int16_t   index_xyz[3];    // indexes to triangle's vertices
    int16_t   index_st[3];     // indexes to vertices' texture coorinates

} MD2Face;

extern unsigned char *md2_loadFile(char *fn);
extern void md2_printHeaderInfo(unsigned char *data);
extern void md2_printSkinNames(unsigned char *data);
extern void md2_printFrameiHeader(unsigned char *data, unsigned int i);


//extern int md2_renderFrameFlat(Shader *s, unsigned char *data, unsigned int i);
extern int md2_renderFrameFlat(Shader *s, unsigned char *data, unsigned int curr_frame);

#endif
