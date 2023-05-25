#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec3 inLightPos;

layout (location = 0) out float outFragColor;

void main() 
{
    vec3 lightVec = inPos.xyz - inLightPos;

    outFragColor = 1.0;//dot(lightVec, lightVec);
}