
#include <stdio.h>
#include <assert.h>

#include "Graphics/image.h"
#include "Graphics/text.h"

#define  WIDTH  512
#define  HEIGHT 512

//#define  LEFT   1.8f
//#define  RIGHT  2.99f
//#define  TOP    1.5f
//#define  BOTTOM 0.0f

//#define  LEFT   2.5f
//#define  RIGHT  2.8f
//#define  BOTTOM 0.9f
//#define  TOP    1.4f

#define  LEFT   2.83f
#define  RIGHT  2.87f
#define  BOTTOM 0.5f
#define  TOP    0.8f


//#define  LEFT   0.0f
//#define  RIGHT  3.0f
//#define  BOTTOM 0.0f
//#define  TOP    3.0f




//#define  POPULATION 0.1f
//#define  FEEDBACK   1.99f

#define INVISIBLE 50 
#define VISIBLE   1000


void setUniversalPoint(Image *img, float xu, float yu, char *color) {
  
  float xs,ys;
  
  xs = (xu - LEFT) * WIDTH / (RIGHT - LEFT);
  ys = (yu - TOP) * HEIGHT / (BOTTOM - TOP); 


 
  /* points are inside the "viewing area" ? */
  if ((unsigned int) xs >= 0 && (unsigned int) xs < WIDTH &&
      (unsigned int) ys >= 0 && (unsigned int) ys < HEIGHT) {
    image_getPixel(img, (unsigned int)xs, (unsigned int)ys,color);
    color[0] = color[0]+5 < 255 ? color[0]+5 : 255; 
    color[1] = color[1]+5 < 255 ? color[1]+5 : 255; 
    color[2] = color[2]+5 < 255 ? color[2]+5 : 255; 
    image_setPixel(img, (unsigned int)xs, (unsigned int)ys,color);
  }
}

float f(float p, float k) {
  return (p + k * p * (1.0f - p));
}


void measlesIteration(Image *img) {
  int range;
  float pop;
  float deltaXPerPixel;
  float feedback;

  char color[3] = {0,0,0};

  deltaXPerPixel = (RIGHT - LEFT) / (float)WIDTH; 
  
  for (range = 0; range < WIDTH; ++range){

   
    feedback = LEFT + range * deltaXPerPixel;
    pop = 0.3;

    for (int i = 0; i < INVISIBLE; ++i) {
      pop = f(pop,feedback);
    }
    for (int i = 0; i < VISIBLE; ++i) {
      setUniversalPoint(img,feedback,pop,color);
      pop = f(pop,feedback);
    }
    
  }

}


void main(void){
  
  Image *out; 

  out = image_create(WIDTH,HEIGHT,3);

  /* create feigenbaum image */
  
  //char color[3] = {255,255,255};
  
  //image_setPixel(out,10,10,color);
  measlesIteration(out);

  /* store image to output file */
  image_storeRaw("feigenbaum.raw",out);
  image_destroy(out);


}
