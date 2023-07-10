#version 450

layout( location = 0 ) in vec3 inPos;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec2 inUV0;
layout( location = 3 ) in vec2 inUV1;
layout( location = 4 ) in vec4 inJoint0;
layout( location = 5 ) in vec4 inWeight0;

layout( push_constant ) uniform PushConsts { layout( offset = 0 ) mat4 mvp; }
pushConsts;

layout( location = 0 ) out vec3 outUVW;

out gl_PerVertex { vec4 gl_Position; };

void main()
{
    outUVW      = inPos;
    gl_Position = pushConsts.mvp * vec4( inPos.xyz, 1.0 );
}
