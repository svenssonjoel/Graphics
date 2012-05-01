#version 140

uniform mat4 proj;
uniform mat4 mod;



in vec4 Vertex; 
in vec2 TexCoord0;

out vec2 TexCoord;


void main() {
  
  TexCoord = TexCoord0;
  gl_Position = proj * (mod * Vertex);
 
}


