#version 450

#extension GL_GOOGLE_include_directive : require

#include "Common/VertexLayout.h"

layout (location = 0) out vec4 outPos;
layout (location = 1) out vec3 outLightPos;

layout (binding = 0) uniform UBO 
{
	mat4 depthMVP;
	vec4 lightPos;
} ubo;

out gl_PerVertex 
{
	vec4 gl_Position;
};
 
void main()
{
	gl_Position = ubo.depthMVP * vec4(inPos, 1.0);

	outPos = vec4(inPos, 1.0);	
	outLightPos = ubo.lightPos.xyz; 
}