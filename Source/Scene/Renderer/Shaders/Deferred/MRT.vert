#version 450

layout( location = 0 ) in vec3 inPos;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec2 inUV0;
layout( location = 3 ) in vec4 inJoint0;
layout( location = 4 ) in vec4 inWeight0;

#define MAX_NUM_LIGHTS 64

struct DirectionalLightData
{
    vec3  Direction;
    vec3  Color;
    float Intensity;
};

struct PointLightData
{
    vec3  WorldPosition;
    vec3  Color;
    float Intensity;
};

struct SpotlightData
{
    vec3  WorldPosition;
    vec3  LookAtDirection;
    vec3  Color;
    float Intensity;
    float Cone;
};

layout( set = 0, binding = 0 ) uniform UBO
{
    mat4 projection;
    mat4 model;
    mat4 view;
    vec3 camPos;

    int                  DirectionalLightCount;
    DirectionalLightData DirectionalLights[MAX_NUM_LIGHTS];

    int           SpotlightCount;
    SpotlightData Spotlights[MAX_NUM_LIGHTS];

    int            PointLightCount;
    PointLightData PointLights[MAX_NUM_LIGHTS];
}
ubo;

layout( location = 0 ) out vec3 outWorldPos;
layout( location = 1 ) out vec3 outNormal;
layout( location = 2 ) out vec2 outUV0;
layout( location = 3 ) out vec2 outUV1;

void main()
{
    gl_Position = ubo.projection * ubo.view * vec4( inPos.xyz, 1.0 );

    outWorldPos = inPos;
    outNormal   = inNormal;
    outUV0      = inUV0;
}
