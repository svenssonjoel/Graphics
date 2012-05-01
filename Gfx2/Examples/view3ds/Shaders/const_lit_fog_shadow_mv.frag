#version 140

uniform vec4 Color;

in vec3 normal;
in vec4 position;
in float fogcoord;
in vec4 projectedCoord;
out vec4 out_FragColor;

uniform vec3 lightPos;
uniform sampler2DShadow shadowTexture;

void main() {
     
  vec4 lightColor = vec4(1.0,1.0,1.0,1.0);
  vec3 lightDir =   lightPos - vec3(position);
  vec3 lightDirNormalized = normalize(lightDir);
  
  float dist = sqrt(lightDir.x * lightDir.x + 
		    lightDir.y * lightDir.y + 
		    lightDir.z * lightDir.z);
  float att = 1.0 / (1.0 + 0.0 * dist + 0.0 * dist * dist); // tweak here
  
  float normalDotLight = max(0.0,dot(normal,lightDirNormalized));
  
  vec4 diffuse = (att * lightColor) * Color * normalDotLight;
  //if (Color.a >= 0.0) 
  //  diffuse = vec4(1.0,1.0,1.0,1.0);

  vec4 ambient = 0.01 * Color;

 
  float fog = (100 - fogcoord) * 0.01;
  fog = clamp(fog,0.0,1.0);
  vec4  fogcolor = vec4(0.5,0.5,0.5,1.0);
    
  

  float shadeFactor = 1.0;

  // vec3 coordPos  = projectedCoord.xyz / projectedCoord.w;
	
  //if(coordPos.x >= 0.0 && coordPos.y >= 0.0 && coordPos.x <= 1.0 && coordPos.y <= 1.0 )
  //{
  //    shadeFactor = texture(shadowTexture, coordPos) < (coordPos.z - 0.0001) ? 0.5 : 1.0;
      //shadeFactor = coordPos.z;
      //if (rValue == 0.5)
      //{
      //  specularItensity = 0.0;
      //}
  //}


  shadeFactor = textureProj(shadowTexture, projectedCoord);
  shadeFactor = shadeFactor * 0.50  + 0.50;
  
  vec4  color = ambient; 
  color += diffuse;
  //if (Color.a > 0.0) {
  //  color = Color;
  //}
  color = vec4(color.rgb *  shadeFactor,1.0);// + ambient; 

  
  out_FragColor = color; // mix (fogcolor,color,fog);
  
}


