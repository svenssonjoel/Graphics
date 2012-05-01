
#ifndef __IMAGE_H
#define __IMAGE_H

typedef struct {
  unsigned int width; 
  unsigned int height;
  unsigned int bpp;    /*Bytes Per Pixel*/
  unsigned char *data;
} Image;





extern int image_storeRaw(char *filename, Image *img);
extern int image_loadRaw(char *filename, Image *img);
extern Image *image_loadPNG(char *filename);

extern Image *image_create(unsigned int w, unsigned int h, unsigned int bpp);
extern void image_destroy(Image *img);

extern Image *image_doubleByteToGrey(Image *test);
extern void image_addAlpha(Image *src, unsigned char *color, Image *dst);

extern void image_clear(Image *img);

extern void image_cpyRegion(Image *src, 
			    unsigned int x, 
			    unsigned int y, 
			    unsigned int w,
			    unsigned int h, 
			    Image *dst,
			    unsigned int xd,
			    unsigned int yd);

extern int image_vFlip(Image *src);
extern int image_hFlip(Image *src);


extern void image_setPixel(Image *dst, 
			   unsigned int pos_x, 
			   unsigned int pos_y, 
			   unsigned char *color);

extern void image_getPixel(Image *dst, 
			   unsigned int pos_x, 
			   unsigned int pos_y, 
			   unsigned char *color);

extern void image_hLine(Image *dst, 
			unsigned int start_x, 
			unsigned int start_y, 
			unsigned int length, 
			unsigned char *color);

extern void image_vLine(Image *dst, 
			unsigned int start_x, 
			unsigned int start_y, 
			unsigned int length, 
			unsigned char *color);


extern void image_drawLine(Image *dst, 
			   int xa, int ya, 
			   int xb, int yb, 
			   unsigned char *color);
extern void image_drawCircle(Image *dst,
			     int xc, 
			     int yc, 
			     int r,
			     unsigned char *color);

extern void image_fillCircle(Image *dst,
			     int xc, 
			     int yc, 
			     int r,
			     unsigned char *color);


extern void image_reduceAnd(Image *src, 
			    unsigned int xs, 
			    unsigned int ys, 
			    unsigned int xe, 
			    unsigned int ye, 
			    unsigned char *result);

extern void image_reduceOr(Image *src, 
			   unsigned int xs, 
			   unsigned int ys, 
			   unsigned int xe, 
			   unsigned int ye, 
			   unsigned char *result);



extern void image_dotTemplate(Image *img, unsigned int bw, unsigned int bh);
extern void image_gridTemplate(Image *img, unsigned int bw, unsigned int bh);

#endif
