#version 140
/* simplest possible texture mapping fragment shader */

in vec2 TexCoord;

out vec4 out_FragColor;

uniform sampler2D tex;

void main() {
     
  vec4 c1 = vec4(texture(tex,TexCoord));   
  //if(c1.a <= 0.9){
    //  discard;
  //} 
  //else {
  out_FragColor = c1; 
  //}
  
}


