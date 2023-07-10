#extension GL_EXT_nonuniform_qualifier : enable

layout( location = 0 ) in vec2 inUV;

layout( set = 1, binding = 0 ) uniform sampler2D samplerPosition;
layout( set = 1, binding = 1 ) uniform sampler2D samplerNormal;
layout( set = 1, binding = 2 ) uniform sampler2D samplerAlbedo;
layout( set = 1, binding = 3 ) uniform sampler2D samplerOcclusionMetalRough;

layout( set = 2, binding = 0 ) uniform sampler2D gDirectionalShadowMaps[];
layout( set = 3, binding = 0 ) uniform sampler2D gSpotlightShadowMaps[];
layout( set = 4, binding = 0 ) uniform samplerCube gPointLightShadowMaps[];


layout( location = 0 ) out vec4 outFragcolor;

#define MAX_NUM_LIGHTS 64

struct DirectionalLightData
{
    vec3  Direction;
    vec3  Color;
    float Intensity;
    mat4  Transform;
    int   IsOn;
};

struct PointLightData
{
    vec3  WorldPosition;
    vec3  Color;
    float Intensity;
    int   IsOn;
};

struct SpotlightData
{
    vec3  WorldPosition;
    vec3  LookAtDirection;
    vec3  Color;
    float Intensity;
    float Cone;
    mat4  Transform;
    int   IsOn;
};

struct sShaderMaterial
{
    vec4 mBaseColorFactor;
    int  mBaseColorTextureID;
    int  mBaseColorUVChannel;

    float mMetallicFactor;
    float mRoughnessFactor;
    int   mMetalnessUVChannel;
    int   mMetalnessTextureID;

    float mOcclusionStrength;
    int   mOcclusionUVChannel;
    int   mOcclusionTextureID;

    vec4 mEmissiveFactor;
    int  mEmissiveTextureID;
    int  mEmissiveUVChannel;

    int mNormalTextureID;
    int mNormalUVChannel;

    float mAlphaThreshold;
};

// Scene bindings
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

layout( set = 0, binding = 1 ) uniform UBOParams
{
    float exposure;
    float gamma;
    float AmbientLightIntensity;
    vec4  AmbientLightColor;
    float debugViewInputs;
    float debugViewEquation;
    float grayscaleRendering;
}
uboParams;

const float PI = 3.14159265359;

struct LightData
{
    // The color (.rgb) and pre-exposed intensity (.w) of the light. The color is an RGB value in the linear sRGB color space.
    // The pre-exposed intensity is the intensity of the light multiplied by the camera's exposure value.
    vec4  mColorIntensity;

    // The normalized light vector, in world space (direction from the current fragment's position to the light).
    vec3  mL;

    // The normalized light half vector, in world space (direction from the current fragment's position to the light).
    vec3  mH;

    // The dot product of the shading normal (with normal mapping applied) and the light vector. This value is equal to the result of
    // saturate(dot(getWorldSpaceNormal(), lightData.l)). This value is always between 0.0 and 1.0. When the value is <= 0.0,
    // the current fragment is not visible from the light and lighting computations can be skipped.
    float mNdotL;

    // The position of the light in world space.
    vec3  mWorldPosition;

    // Attenuation of the light based on the distance from the current fragment to the light in world space. This value between 0.0 and 1.0
    // is computed differently for each type of light (it's always 1.0 for directional lights).
    float mAttenuation;

    // Visibility factor computed from shadow maps or other occlusion data specific to the light being evaluated. This value is between 0.0 and
    // 1.0.
    float mVisibility;    
};

const int enablePCF = 1;

