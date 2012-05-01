#version 140

uniform mat4 proj;


in vec4 Color;
in vec4 Vertex; 

smooth out vec4 SmoothColor;

void main() {
  
  SmoothColor = Color;
  gl_Position = proj * Vertex;
}


