#version 150

uniform mat4 proj;
uniform mat4 mod;
uniform mat3 normalMatrix;

uniform mat4 shadowMatrix;

in vec4 Vertex; 
in vec3 Normal;

out vec3 normal;
out vec4 position;
out float fogcoord;
out vec4  projectedCoord;

void main() {
  
  normal = normalize(normalMatrix * Normal);
  gl_Position = proj * (mod * Vertex);
  position = mod * Vertex;
  fogcoord = abs(position.z); 
  
  projectedCoord =  shadowMatrix*Vertex;
}


