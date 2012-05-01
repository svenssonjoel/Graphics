#version 140

smooth in vec4 SmoothColor;
out vec4 out_FragColor;


void main() {
  out_FragColor = SmoothColor;
}


