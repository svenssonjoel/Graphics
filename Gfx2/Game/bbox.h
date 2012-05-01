/* bounding boxes  */

#ifndef __BBOX_H_
#define __BBOX_H_

#include "point.h"
#include "../Graphics/image.h"

typedef struct {
  Point vertices[4];
}BBox;


void bbox_fromImage(Image *img, BBox *bb); 

void bbox_drawToImage(BBox *bb, 
		      unsigned int x, 
		      unsigned int y,
		      unsigned char *color,
		      Image *image);


#endif 
