#version 140


in vec2 TexCoord;

out vec4 out_FragColor;

uniform sampler2D tex;


/* extremely simple texture mapping */

void main() {
     
 
  vec4 c1 =  vec4(texture(tex,TexCoord));   
  
  if (c1.a == 1.0)
    out_FragColor = c1;   
  else discard;
}


