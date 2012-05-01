
#include <stdio.h>

#include "image.h"
#include "text.h"

void main(void){
  
  Image *img;
  Image *tmpl; 
 
  Image *out; 

  CharSet *whiteChars;
  
  //tmpl = image_create(512,512,4);
  //img = image_create(512,512,4);
  //image_loadRaw("test.raw",img);
  //image_storeRaw("test2.raw",img);
  
   
  //image_template(tmpl,32,32);
  //image_storeRaw("templ.raw",tmpl);

  //  image_template(tmpl,128,128);
  //image_storeRaw("templ2.raw",tmpl);

  
  whiteChars = text_loadCharSet("ASCIIWhite.raw",32,32);
  
  img = image_create(512,512,4);
  image_loadRaw("ASCIIWhite.raw",img);

  out = image_create(512,512,4);
  
  //image_cpyRegion(img,32,4*32,32,32,out,0,0);
  //image_cpyRegion(img,0,5*32,32,32,out,32,0);
  //image_cpyRegion(img,32,4*32,32,32,out,64,0);

  //image_cpyRegion(whiteChars->image,32,4*32,32,32,out,0,0);
  //image_cpyRegion(whiteChars->image,0,5*32,32,32,out,32,0);
  //image_cpyRegion(whiteChars->image,32,4*32,32,32,out,64,0);
  
  text_putCh(whiteChars,'A',0,1,out);
  text_putCh(whiteChars,'P',1,1,out);
  text_putCh(whiteChars,'A',2,1,out);
 
  text_putStr(whiteChars,"Monkey!",0,2,out);
  

  image_storeRaw("output.raw",out);
  text_destroyCharSet(whiteChars);
  image_destroy(out);
  image_destroy(img);

}
