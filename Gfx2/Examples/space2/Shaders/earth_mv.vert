#version 140

uniform mat4 proj;
uniform mat4 mod;
uniform mat3 normalMatrix;


in vec4 Vertex; 
in vec2 TexCoord0;
in vec3 Normal;

out vec2 TexCoord;
out vec3 normal;
out vec4 position;

void main() {
  
  normal = normalize(normalMatrix * Normal);
  TexCoord = TexCoord0;
  gl_Position = proj * (mod * Vertex);
  position = mod * Vertex;
 
}


