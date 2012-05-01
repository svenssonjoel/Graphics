
#ifndef __IMAGE_H
#define __IMAGE_H

typedef struct {
  unsigned short width;
  unsigned short height;
  unsigned short bpp;
  unsigned char *data;
} Image;



extern void image_template(Image *img, unsigned short bw, unsigned short bh);

extern int image_storeRaw(char *filename, Image *img);
extern int image_loadRaw(char *filename, Image *img);
extern Image* image_create(unsigned short w, unsigned short h, unsigned short bpp);
extern void image_destroy(Image *img);
extern void image_cpyRegion(Image *src, 
			    unsigned short x, 
			    unsigned short y, 
			    unsigned short w,
			    unsigned short h, 
			    Image *dst,
			    unsigned short xd,
			    unsigned short yd);
#endif
