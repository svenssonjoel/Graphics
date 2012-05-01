/* 

  images 
  
  loading, storing. 


  step1: raw image data

*/

#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>


Image* image_create(unsigned short w, unsigned short h, unsigned short bpp){
  
  Image *new; 

  if (!(new = malloc(sizeof(Image)))) {
      return NULL;
  }
   
  new->width = w; 
  new->height = h; 
  new->bpp = bpp;
  new->data = NULL;

  if (!(new->data = malloc(w*h*bpp))){
    return NULL;
  }

  return new;
} 

void image_destroy(Image *img){
  if (img) {
    if (img->data)
      free(img->data);
    
    free(img);
  }
}

int image_loadRaw(char *filename, Image *img){

  FILE *fp; 
  
  

  unsigned int bytes = img->width * img->height * img->bpp; // bytes to read

  if (!(fp = fopen(filename,"r"))) 
    return -1;

   
  fread((void*)img->data,1,bytes,fp);
  
  fclose(fp);
 
}

int image_storeRaw(char *filename, Image *img) {
  FILE *fp; 
  
  unsigned int bytes =   img->width * img->height * img->bpp;
  

  if (!(fp = fopen(filename,"w"))) 
    return -1;

  fwrite((void*)img->data,1,bytes,fp);
  
  fclose(fp);

}

void image_template(Image *img, unsigned short bw, unsigned short bh){
  
  unsigned short w = img->width;
  unsigned short h = img->height;
  unsigned short bpp = img->bpp; 

  int cols = w / bw;
  int rows = h / bh;

  int step_w = bw*bpp;
  int step_h = bh*(step_w * cols);

  memset(img->data,0,w*h*bpp);

  for (int r = 0; r < rows; r ++) {
      for (int c = 0; c < cols; c ++) {
	assert (r * bh < h); 
	assert (c * bw < w);

	memset(img->data + (r*step_h) + (c * step_w),0xFF,4);
    }
  }
      
}

//cpy part of image into image, same format required
void image_cpyRegion(Image *src, 
		     unsigned short x, 
		     unsigned short y, 
		     unsigned short w,
		     unsigned short h, 
		     Image *dst,
		     unsigned short xd,
		     unsigned short yd){

  int lines = h;
  int bytesPerLine = w * src->bpp;

  for (int i = 0; i < lines; i ++) {
    int src_offset = (((y+i)*src->width) + x) * src->bpp;
    int dst_offset = (((yd+i)*dst->width) + xd) * dst->bpp; //same bpp is required
    memcpy(dst->data + dst_offset,src->data + src_offset, bytesPerLine);
  }
   
  



}
