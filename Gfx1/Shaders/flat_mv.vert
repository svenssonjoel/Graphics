#version 140

uniform mat4 proj;
uniform mat4 mod;


in vec4 Color;
in vec4 Vertex; 

smooth out vec4 SmoothColor;

void main() {
  
  SmoothColor = Color;
  gl_Position = proj * (mod * Vertex);
  //gl_Position = mod * proj * Vertex;	
}


