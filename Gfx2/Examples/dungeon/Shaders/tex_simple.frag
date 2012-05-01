#version 140


in vec2 TexCoord;

out vec4 out_FragColor;

uniform sampler2D tex;


/* extremely simple texture mapping */

void main() {
     
  vec4 c0 = vec4(texture(tex,TexCoord));   
  //vec4 c1 = vec4(0.1,0.1,0.1,1.0);
  out_FragColor = c0;   
}


