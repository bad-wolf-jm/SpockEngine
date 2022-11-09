#version 450

layout( location = 0 ) in vec3 inPos;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec2 inUV0;
layout( location = 3 ) in vec2 inUV1;
layout( location = 4 ) in vec4 inJoint0;
layout( location = 5 ) in vec4 inWeight0;

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

#define MAX_NUM_JOINTS 256

layout( set = 2, binding = 0 ) uniform UBONode
{
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
    float jointCount;
}
node;
// layout (binding = 0) uniform UBO
// {
//     mat4 projection;
//     mat4 model;
//     mat4 view;
//     vec4 instancePos[3];
// } ubo;

layout( location = 0 ) out vec3 outWorldPos;
layout( location = 1 ) out vec3 outNormal;
layout( location = 2 ) out vec2 outUV0;
layout( location = 3 ) out vec2 outUV1;

// layout( location = 0 ) out vec3 outNormal;
// layout( location = 1 ) out vec2 outUV;
// layout( location = 2 ) out vec3 outColor;
// layout( location = 3 ) out vec3 outWorldPos;
// layout( location = 4 ) out vec3 outTangent;

void main()
{
    // vec4 tmpPos = vec4( inPos.xyz, 1.0 ) + ubo.instancePos[gl_InstanceIndex];

    gl_Position = ubo.projection * ubo.view * vec4( inPos.xyz, 1.0 );

    outUV0 = inUV0;
    outUV1 = inUV1;

    // Vertex position in world space
    outWorldPos = vec3( inPos );

    // Normal in world space
    mat3 mNormal = transpose( inverse( mat3( ubo.model ) ) );
    outNormal    = mNormal * normalize( inNormal );

    // outTangent   = mNormal * normalize( inTangent.xz );
    // Currently just vertex color
    // outColor = inColor;
}
