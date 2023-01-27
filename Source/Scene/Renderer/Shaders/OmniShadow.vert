#version 450

layout (location = 0 ) in vec3 inPos;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec2 inUV0;
layout( location = 3 ) in vec4 inJoint0;
layout( location = 4 ) in vec4 inWeight0;

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