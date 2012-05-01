#version 140


in vec2 TexCoord;
in vec3 normal;
in vec4 position;

out vec4 out_FragColor;

uniform sampler2D tex;
uniform vec3 lightPos;

void main() {
     
  vec4 lightColor = vec4(1.0,1.0,1.0,1.0);
  vec3 lightDir =   lightPos - vec3(position);
  vec3 lightDirNormalized = normalize(lightDir);
  
  float dist = sqrt(lightDir.x * lightDir.x + 
		    lightDir.y * lightDir.y + 
		    lightDir.z * lightDir.z);
  float att = 1.0 / (1.0 + 0.0 * dist + 0.0 * dist * dist); // tweak here
  
  float normalDotLight = max(0.0,dot(normal,lightDirNormalized));

  vec4 c1 = vec4(texture(tex,TexCoord));   
  
  vec4 diffuse = (att * lightColor) * c1 * normalDotLight;
  
  vec4 ambient = 0.1 * c1;

  
  
  
  out_FragColor = diffuse + ambient; 

  
}


