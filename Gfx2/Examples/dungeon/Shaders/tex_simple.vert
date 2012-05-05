#version 140

in vec3 Vertex; 
in vec2 TexCoord0;

out vec2 TexCoord;

uniform mat4 proj;
uniform mat4 mod;

void main() {
  
  TexCoord = TexCoord0;
  gl_Position = proj * (mod * vec4(Vertex,1.0));
 
}


