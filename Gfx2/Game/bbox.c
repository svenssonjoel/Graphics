
#include "bbox.h"


void bbox_fromImage(Image *img, BBox *bb) {
  
  bb->vertices[0].x = 0; 
  bb->vertices[0].y = 0; 
  
  bb->vertices[1].x = 0; 
  bb->vertices[1].y = img->height; 
  
  bb->vertices[2].x = img->width; 
  bb->vertices[2].y = img->height; 
  
  bb->vertices[3].x = img->width; 
  bb->vertices[3].y = 0; 
}

void bbox_drawToImage(BBox *bb, 
		      unsigned int x, 
		      unsigned int y,
		      unsigned char *color,
		      Image *dst){
  

  image_vLine(dst, 
	      x + bb->vertices[0].x,
	      y + bb->vertices[0].y,
	      bb->vertices[1].y,
	      color);
 

  image_vLine(dst, 
	      x + bb->vertices[3].x,
	      y + bb->vertices[3].y,
	      bb->vertices[2].y,
	      color);



  image_hLine(dst, 
	      x + bb->vertices[0].x,
	      y + bb->vertices[0].y,
	      bb->vertices[3].x,
	      color);
 

  image_hLine(dst, 
	      x + bb->vertices[1].x,
	      y + bb->vertices[1].y,
	      bb->vertices[2].x,
	      color);
}
