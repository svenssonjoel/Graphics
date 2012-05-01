


#include "text.h"
#include <stdlib.h>
#include <stdio.h>

/* Load a 256 character set*/
CharSet *text_loadCharSet(char *filename, unsigned short width, unsigned short height){

  CharSet *new; 

  new = malloc(sizeof(CharSet));
  if (!new) return NULL;

  new->image = image_create(512,512,4);
  if (image_loadRaw(filename,new->image) == -1)  {
    fprintf(stderr,"error opening file\n");
    exit(-1);
  }
  
  
  new->char_width = width;
  new->char_height = height;
  
  return new;
}

void text_destroyCharSet(CharSet *cs) {
  if (cs) {
    if (cs->image) 
      free(cs->image);
    free(cs);
  }
  
}

void text_putCh(CharSet *cs, char c,int x, int y, Image *where) {
  
  int charsPerRow = cs->image->width / cs->char_width;
  int row = c / charsPerRow;
  int col = c % charsPerRow; 
    
  image_cpyRegion(cs->image,
		  col*cs->char_width,
		  row*cs->char_height,
		  cs->char_width,
		  cs->char_height,
		  where,
		  x*cs->char_width,
		  y*cs->char_height);

}

void text_putStr(CharSet *cs,char *str,int x, int y, Image *where){
  int i = 0; 
  while (str[i]) {
    text_putCh(cs,str[i],x+i,y,where);
    i ++;
  }
}
