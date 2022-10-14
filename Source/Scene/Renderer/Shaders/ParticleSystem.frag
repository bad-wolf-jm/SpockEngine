#version 450

// Interpolated values from the vertex shaders
layout( location = 0 ) in vec2 UV;
layout( location = 1 ) in vec4 particlecolor;

// Ouput data
layout( location = 0 ) out vec4 color;

void main()
{
    // Output color = color of the texture at the specified UV
    // color = texture( myTextureSampler, UV ) * particlecolor;
    color = particlecolor;
}