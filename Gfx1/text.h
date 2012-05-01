
#ifndef __TEXT_H
#define __TEXT_H

#include "image.h"


typedef struct {
  unsigned short char_width;
  unsigned short char_height;
  Image *image;
} CharSet;
  
extern CharSet *text_loadCharSet(char *filename, unsigned short width, unsigned short height);
extern void text_putCh(CharSet *cs, char c,int x, int y, Image *where);
extern void text_putStr(CharSet *cs,char *str,int x, int y, Image *where);
extern void text_destroyCharSet(CharSet *cs);


#endif
