#version 150

uniform mat4 proj;
uniform mat4 mod;

in vec4 Vertex;

void main(void) {
  gl_Position = proj*(mod*Vertex);
}
