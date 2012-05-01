#version 140

uniform mat4 proj;
uniform mat4 mod;


in vec4 Vertex; 

void main() {
  
  gl_Position = proj * (mod * Vertex);
 
}


