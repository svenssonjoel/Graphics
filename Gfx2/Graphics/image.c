/* 

  images 
  
  loading, storing. 


  step1: raw image data
  step2: png image data



*/

#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include <math.h>

#include <png.h>

/* TODO 
   

   Add error handling to many of the routines. 
   Add Clipping to some,.... (where to add this is a bit open) 
      (maybe provide a safe and an unsafe version of routines) 

 */ 





// -----------------------------------------------------------------------------
// Image Routines
// -----------------------------------------------------------------------------

Image* image_create(unsigned int w, unsigned int h, unsigned int bpp){
  
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

void image_clear(Image *img) {
  if (img) {
    memset(img->data,0,img->width * img->height * img->bpp);
  }
  /* should this return an error if there is no "img" ?  */
}

// -----------------------------------------------------------------------------
// Image Loading and storing
// -----------------------------------------------------------------------------
int image_loadRaw(char *filename, Image *img){

  if (!img) return -1;
  
  FILE *fp; 
  
  unsigned int bytes = img->width * img->height * img->bpp; // bytes to read

  if (!(fp = fopen(filename,"r"))) 
    return -1;

   
  /* Read data. Check if correct amount is recieved */
  if (fread((void*)img->data,1,bytes,fp) != bytes) {
    fclose(fp);
    return -1;
  }
  
  fclose(fp);
  return 0; 
}

int image_storeRaw(char *filename, Image *img) {

  if (!img) return -1;
  
  FILE *fp; 
  
  unsigned int bytes =   img->width * img->height * img->bpp;
  

  if (!(fp = fopen(filename,"w"))) 
    return -1;

  
  if (fwrite((void*)img->data,1,bytes,fp) != bytes) {
    fclose(fp);
    return -1; 
  }
  
  fclose(fp);
  return 0;

}
// -----------------------------------------------------------------------------
// convert to greyscale
// -----------------------------------------------------------------------------
Image *image_doubleByteToGrey(Image *src) {
  Image *tmp; 
  
  tmp = image_create(src->width,src->height,3);
  
  unsigned int num_elems = src->width * src->height; 
  
  short val; 
  float dist; 
  
  for (int i = 0; i < num_elems; i++) {
    val = src->data[i*2];
    val += src->data[i*2+1] << 8;
    
    dist = (float)val / 2000; 
    tmp->data[i*3] = dist * 255;
    tmp->data[i*3+1] = dist * 255;
    tmp->data[i*3+2] = dist * 255;
    
      
  }

  return tmp;
  
}

// -----------------------------------------------------------------------------
// Image Add alpha channel
// -----------------------------------------------------------------------------
void image_addAlpha(Image *src, unsigned char *color, Image *dst) {
  /* requirements. 
     src is RGB
     dst is RGBA 
     dimensions are equal 
     no error checking performed. (ADD) 
  */

  unsigned int num_elems = src->width * src->height; 

  for (int i = 0; i < num_elems; ++i) {
    
    if (src->data[i*3] == color[0] && 
	src->data[i*3+1] == color[1] && 
	src->data[i*3+2] == color[2]) {
      dst->data[i*4] = 0; 
      dst->data[i*4+1] = 0; 
      dst->data[i*4+2] = 0; 
      dst->data[i*4+3] = 0; 
    }
    else {
      dst->data[i*4] =   src->data[i*3];  
      dst->data[i*4+1] = src->data[i*3+1]; 
      dst->data[i*4+2] = src->data[i*3+2]; 
      dst->data[i*4+3] = 255;     
    }
    
  }
}




// -----------------------------------------------------------------------------
// COPY regions
// -----------------------------------------------------------------------------
//cpy part of image into image, same format required
void image_cpyRegion(Image *src, 
		     unsigned int x, 
		     unsigned int y, 
		     unsigned int w,
		     unsigned int h, 
		     Image *dst,
		     unsigned int xd,
		     unsigned int yd){

  int lines = h;
  int bytesPerLine = w * src->bpp;

  for (int i = 0; i < lines; i ++) {
    int src_offset = (((y+i)*src->width) + x) * src->bpp;
    int dst_offset = (((yd+i)*dst->width) + xd) * dst->bpp; //same bpp is required
    memcpy(dst->data + dst_offset,src->data + src_offset, bytesPerLine);
  }
}



int image_vFlip(Image *src) {

  unsigned char *row = NULL;

  int rows = src->height;
  int bpr  = src->bpp * src->width;
  
  row = (unsigned char *)malloc(bpr); 
  if (!row) return -1;
  
  for (int i = 0; i < rows; ++i) {
    memcpy(row,src->data + (i * bpr),bpr);
    for (int p = 0; p < src->width; ++p) {
      memcpy(src->data + (i * bpr) + ((src->width - p) * src->bpp),row + (p * src->bpp),src->bpp);
    }
    
  }
  free(row);
  return 0;

}

int image_hFlip(Image *src) {

  unsigned char *r1 = NULL;

  int rows = src->height;
  int bpr  = src->bpp * src->width;
  
  r1 = (unsigned char *)malloc(bpr); 
  if (!r1) return -1;
  
  for (int i = 0; i < rows / 2; ++i) {
    memcpy(r1,src->data + ((rows - i - 1) * bpr),bpr);
    memcpy(src->data + ((rows - i - 1) * bpr), 
	   src->data + (i * bpr),bpr);
    memcpy(src->data + (i * bpr),
	   r1,bpr);
  }
  free(r1);
  return 0;

}

		 
		 



// -----------------------------------------------------------------------------
// getPixel and putPixel
// -----------------------------------------------------------------------------
void image_setPixel(Image *dst, 
		    unsigned int pos_x, 
		    unsigned int pos_y, 
		    unsigned char *color)  {

  unsigned int mem_loc = (pos_x + (dst->width * pos_y)) * dst->bpp; /* distance in #bytes */

  memcpy(dst->data + mem_loc, color, dst->bpp);

}

void image_getPixel(Image *dst, 
		    unsigned int pos_x, 
		    unsigned int pos_y, 
		    unsigned char *color)  {

  unsigned int mem_loc = (pos_x + (dst->width * pos_y)) * dst->bpp; /* distance in #bytes */

  memcpy(color, dst->data + mem_loc, dst->bpp);

}


// -----------------------------------------------------------------------------
//    Horizontal and vertical lines
// -----------------------------------------------------------------------------
void image_hLine(Image *dst, 
		 unsigned int start_x, 
		 unsigned int start_y, 
		 unsigned int length, 
		 unsigned char *color){
 
  /* the generality of this horizontal linedrawing algorithm will 
     make it less efficient. 
     (because of how "color" is handled) 
  */

  unsigned int px = start_x;
  unsigned int py = start_y;
  unsigned int mem_loc; 
  

  if (start_x > dst->width) return; 
  if (start_y < 0) return;
  if (start_y > dst->height) return;
  if (start_x < 0) start_x = 0; 

  unsigned int clipped_length = 
    start_x + length < dst->width ? 
    length : 
    dst->width - start_x;  

  for (int i = 0; i < clipped_length; ++i){
    mem_loc = ((px + i) + (dst->width * py)) * dst->bpp;
    memcpy(dst->data + mem_loc, color, dst->bpp);

  }
}

void image_vLine(Image *dst, 
		 unsigned int start_x, 
		 unsigned int start_y, 
		 unsigned int length, 
		 unsigned char *color){
 
  unsigned int px = start_x;
  unsigned int py = start_y;
  unsigned int mem_loc; 

  if (start_x < 0) return; 
  if (start_x > dst->width) return; 
  if (start_y > dst->height) return;
  if (start_y < 0) start_y = 0;


  unsigned int clipped_length = 
    start_y + length < dst->height ? 
    length : 
    dst->height - start_y;  

  for (int i = 0; i < clipped_length; ++i){
    mem_loc = (px + (dst->width * (py + i))) * dst->bpp;
    memcpy(dst->data + mem_loc, color, dst->bpp);

  }
}


// -----------------------------------------------------------------------------
//  general line
// -----------------------------------------------------------------------------
void image_drawLine(Image *dst, 
		    int xa, int ya, 
		    int xb, int yb, 
		    unsigned char *color){
  
  
  int x_round,y_round;
  int dy = (yb - ya);
  int dx = (xb - xa);
  int steps = dx > dy ? dx : dy; 
    
  float inc_x = (float)dx / steps;
  float inc_y = (float)dy / steps;
    
  float x = xa;
  float y = ya;
  
  unsigned int mem_loc; 

  for (int i=0; i<=steps; ++i) {
    y_round = round(y);
    x_round = round(x);
    mem_loc = (x_round + (dst->width * y_round)) * dst->bpp; 
    memcpy(dst->data + mem_loc, color, dst->bpp);
    y = y + inc_y;
    x = x + inc_x;
  }
}

// -----------------------------------------------------------------------------
//  Circle drawing
// -----------------------------------------------------------------------------
void image_drawCircle(Image *dst,
		      int xc, 
		      int yc, 
		      int r,
		      unsigned char *color) {
  int x = 0;
  int y = r; 
  int e = 0; 
  unsigned int mem_loc;

  mem_loc = ((xc + x) + (dst->width * (yc + y))) * dst->bpp; 
  memcpy(dst->data + mem_loc, color, dst->bpp);
  mem_loc = ((xc + y) + (dst->width * (yc + x))) * dst->bpp; 
  memcpy(dst->data + mem_loc, color, dst->bpp);
    
  mem_loc = ((xc - x) + (dst->width * (yc - y))) * dst->bpp; 
  memcpy(dst->data + mem_loc, color, dst->bpp);
  mem_loc = ((xc - y) + (dst->width * (yc - x))) * dst->bpp; 
  memcpy(dst->data + mem_loc, color, dst->bpp);
  
  while (x <= y) {
    if (e < 0) {
      e = e + y + y - 1; 
      y = y - 1;
    }
    e = e - x - x - 1; 
    x = x + 1; 
    mem_loc = ((xc + x) + (dst->width * (yc + y))) * dst->bpp; 
    memcpy(dst->data + mem_loc, color, dst->bpp);
    mem_loc = ((xc + y) + (dst->width * (yc + x))) * dst->bpp; 
    memcpy(dst->data + mem_loc, color, dst->bpp);
    
    mem_loc = ((xc - x) + (dst->width * (yc + y))) * dst->bpp; 
    memcpy(dst->data + mem_loc, color, dst->bpp);
    mem_loc = ((xc - y) + (dst->width * (yc + x))) * dst->bpp; 
    memcpy(dst->data + mem_loc, color, dst->bpp);

    mem_loc = ((xc + x) + (dst->width * (yc - y))) * dst->bpp; 
    memcpy(dst->data + mem_loc, color, dst->bpp);
    mem_loc = ((xc + y) + (dst->width * (yc - x))) * dst->bpp; 
    memcpy(dst->data + mem_loc, color, dst->bpp);
    
    mem_loc = ((xc - x) + (dst->width * (yc - y))) * dst->bpp; 
    memcpy(dst->data + mem_loc, color, dst->bpp);
    mem_loc = ((xc - y) + (dst->width * (yc - x))) * dst->bpp; 
    memcpy(dst->data + mem_loc, color, dst->bpp);
  }

}

void image_fillCircle(Image *dst,
		      int xc, 
		      int yc, 
		      int r,
		      unsigned char *color) {
  
  int x = 0;
  int y = r; 
  int e = 0; 

  image_hLine(dst,xc-y,yc-x,(xc + y) - (xc - y), color);
  
  while (x <= y) { 
    if (e < 0) {
      e = e + y + y - 1; 
      y = y - 1;
    }
    e = e - x - x - 1; 
    x = x + 1; 
    
    image_hLine(dst,xc-x,yc+y,(xc + x) - (xc - x), color);
    image_hLine(dst,xc-y,yc+x,(xc + y) - (xc - y), color);
  
    image_hLine(dst,xc-x,yc-y,(xc + x) - (xc - x), color);
    image_hLine(dst,xc-y,yc-x,(xc + y) - (xc - y), color);
  }

}


// -----------------------------------------------------------------------------
//  and & or reductions 
// -----------------------------------------------------------------------------

void image_reduceAnd(Image *src, 
		     unsigned int xs, 
		     unsigned int ys, 
		     unsigned int xe, 
		     unsigned int ye, 
		     unsigned char *result) {

  if (xe < 0) return;
  if (ye < 0) return;
  if (xs > src->width) return;
  if (ys > src->width) return;
  
  if (xs < 0) xs = 0;
  if (ys < 0) ys = 0;
  if (xe > src->width) xe = src->width;
  if (ye > src->height) ye = src->height;
  
  
  

  unsigned int lx = xe - xs; 
  unsigned int ly = ye - ys; 
  
  for (int j = 0; j <= ly; ++j) {
    for (int i = 0; i <= lx; ++i) {
      for (int bytespp = 0; bytespp < src->bpp; ++bytespp) {
	result[bytespp] = result[bytespp] && 
	  src->data[(xs + i) + (src->width * (ys + j)) * src->bpp + bytespp];
      }
    }
  }
}



void image_reduceOr(Image *src, 
		     unsigned int xs, 
		     unsigned int ys, 
		     unsigned int xe, 
		     unsigned int ye, 
		     unsigned char *result) {

  /* 
     xs,ys,xe,ye must specify a rectangle within the image.
     up to the caller to make sure.
  */

  unsigned int lx = xe - xs; 
  unsigned int ly = ye - ys; 
  
  for (int j = 0; j <= ly; ++j) {
    for (int i = 0; i <= lx; ++i) {
      for (int bytespp = 0; bytespp < src->bpp; ++bytespp) {
	result[bytespp] = result[bytespp] || 
	  src->data[((xs + i) + (src->width * (ys + j))) * src->bpp + bytespp];
      }
    }
  }
}







// -----------------------------------------------------------------------------
// Create Image Templates
// -----------------------------------------------------------------------------
/* image_template 
   Takes an image and Creates an template image with 
   a grid of specified dimensions
*/
void image_dotTemplate(Image *img, unsigned int bw, unsigned int bh){
  
  unsigned int w = img->width;
  unsigned int h = img->height;
  unsigned int bpp = img->bpp; 

  int cols = w / bw;
  int rows = h / bh;

  int step_w = bw*bpp;
  int step_h = bh*(step_w * cols);

  memset(img->data,0,w*h*bpp);

  for (int r = 0; r < rows; r ++) {
      for (int c = 0; c < cols; c ++) {
	assert (r * bh < h); 
	assert (c * bw < w);

	
	memset(img->data + (r*step_h) + (c * step_w),0xFF,img->bpp);
    }
  }
}


void image_gridTemplate(Image *img, unsigned int bw, unsigned int bh){
  unsigned int w = img->width;
  unsigned int h = img->height;
  unsigned int bpp = img->bpp; 

  int cols = w / bw;
  int rows = h / bh;
  
  unsigned char *color; 
  color = malloc(bpp);
  memset(color,0xFF,bpp); 
  
  for (int i = 1; i < cols; ++i) {
    image_vLine(img,i*bw,0,h,color);
  }
  for (int i = 1; i < rows; ++i) {
    image_hLine(img,0,i*bh,w,color);
  }

  free(color);
}



// -----------------------------------------------------------------------------
// Load a PNG file 
// -----------------------------------------------------------------------------


Image *image_loadPNG(char *filename){
  
  
  
  png_structp png_ptr;
  png_infop info_ptr;
  int number_of_passes;

  unsigned char header[8];


  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr,"image_loadPNG: File not found\n");
    exit(EXIT_FAILURE);
  }

  fread(header, 1, 8, fp);
  if (png_sig_cmp(header, 0, 8)){
    fprintf(stderr,"image_loadPNG: Not a valid PNG file\n");
    exit(EXIT_FAILURE);
  }

  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	
  if (!png_ptr) {
    fprintf(stderr,"image_loadPNG: Memory allocation error\n");
    exit(EXIT_FAILURE);
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr){
    fprintf(stderr,"image_loadPNG: Memory allocation error\n");
    exit(EXIT_FAILURE);
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
   fprintf(stderr,"image_loadPNG: Error\n");
    exit(EXIT_FAILURE);
  }
  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  int width = info_ptr->width;
  int height = info_ptr->height;
  int color_type = info_ptr->color_type;
  //int bit_depth = info_ptr->bit_depth;
  int bpp = 0;
  Image *new; 

  
  number_of_passes = png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  
  //fprintf(stdout," %d %d \n",width,height);
  if (color_type == PNG_COLOR_TYPE_RGB) {
    //  fprintf(stdout,"RGB Color\n");
    bpp = 3;
  }

  if (color_type == PNG_COLOR_TYPE_RGBA) {
    // fprintf(stdout,"RGBA Color\n");
    bpp = 4;
  }
  
  if (bpp != 3 && bpp != 4) {
    fprintf(stderr,"Picture format not supported\n");
    exit(EXIT_FAILURE);
  }
    

  /* rowpointers */
  png_bytep *rowpointers = (png_bytep *)malloc(height * sizeof(png_bytep));
  
  /* allocate space in the image for the image */
  new = image_create(width,height,bpp);
  
  /*setup rowpointers */ 
  for (int i = 0; i < height; i ++) {
    rowpointers[i] = new->data + (i*width*bpp);
  }
  
  //if (setjmp(png_jmpbuf(png_ptr)))
  //  abort_("[read_png_file] Error during read_image");

  //row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
  //for (y=0; y<height; y++)
  //  row_pointers[y] = (png_byte*) malloc(info_ptr->rowbytes);

  png_read_image(png_ptr, rowpointers);

  fclose(fp);
  
  return new;
  

}
 
