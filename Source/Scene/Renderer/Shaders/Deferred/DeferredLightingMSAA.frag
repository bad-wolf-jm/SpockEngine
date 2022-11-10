#version 450

layout (location = 0) in vec2 inUV;

layout (set = 1, binding = 0) uniform sampler2DMS samplerPosition;
layout (set = 1, binding = 1) uniform sampler2DMS samplerNormal;
layout (set = 1, binding = 2) uniform sampler2DMS samplerAlbedo;
layout (set = 1, binding = 3) uniform sampler2DMS samplerOcclusionMetalRough;

layout (location = 0) out vec4 outFragcolor;


#define MAX_NUM_LIGHTS 64

struct DirectionalLightData
{
    vec3 Direction;
    vec3 Color;
    float Intensity;
};

struct PointLightData
{
    vec3 WorldPosition;
    vec3 Color;
    float Intensity;
};

struct SpotlightData
{
    vec3 WorldPosition;
    vec3 LookAtDirection;
    vec3 Color;
    float Intensity;
    float Cone;
};

struct sShaderMaterial
{
    vec4 mBaseColorFactor;
    int mBaseColorTextureID;
    int mBaseColorUVChannel;

    float mMetallicFactor;
    float mRoughnessFactor;
    int mMetalnessUVChannel;
    int mMetalnessTextureID;

    float mOcclusionStrength;
    int mOcclusionUVChannel;
    int mOcclusionTextureID;

    vec4 mEmissiveFactor;
    int mEmissiveTextureID;
    int mEmissiveUVChannel;

    int mNormalTextureID;
    int mNormalUVChannel;

    float mAlphaThreshold;
};

// Scene bindings
layout( set = 0, binding = 0 ) uniform UBO0
{
    mat4 projection;
    mat4 model;
    mat4 view;
    vec3 camPos;

    int DirectionalLightCount;
    DirectionalLightData DirectionalLights[MAX_NUM_LIGHTS];

    int SpotlightCount;
    SpotlightData Spotlights[MAX_NUM_LIGHTS];

    int PointLightCount;
    PointLightData PointLights[MAX_NUM_LIGHTS];
}
ubo0;

layout( set = 0, binding = 1 ) uniform UBOParams
{
    float exposure;
    float gamma;
    float AmbientLightIntensity;
    vec4 AmbientLightColor;
    float debugViewInputs;
    float debugViewEquation;
}
uboParams;


// struct Light {
//     vec4 position;
//     vec3 color;
//     float radius;
// };

// layout (binding = 4) uniform UBO 
// {
//     Light lights[6];
//     vec4 viewPos;
//     int debugDisplayTarget;
// } ubo;

layout (constant_id = 0) const int NUM_SAMPLES = 8;

#define NUM_LIGHTS 6

// Manual resolve for MSAA samples 
vec4 resolve(sampler2DMS tex, ivec2 uv)
{
    vec4 result = vec4(0.0);       
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        vec4 val = texelFetch(tex, uv, i); 
        result += val;
    }    
    // Average resolved samples
    return result / float(NUM_SAMPLES);
}

vec3 calculateLighting(vec3 pos, vec3 normal, vec4 albedo)
{
    vec3 result = vec3(0.0);

    // for(int i = 0; i < NUM_LIGHTS; ++i)
    // {
    //     // Vector to light
    //     vec3 L = ubo.lights[i].position.xyz - pos;
    //     // Distance from light to fragment position
    //     float dist = length(L);

    //     // Viewer to fragment
    //     vec3 V = ubo.viewPos.xyz - pos;
    //     V = normalize(V);
        
    //     // Light to fragment
    //     L = normalize(L);

    //     // Attenuation
    //     float atten = ubo.lights[i].radius / (pow(dist, 2.0) + 1.0);

    //     // Diffuse part
    //     vec3 N = normalize(normal);
    //     float NdotL = max(0.0, dot(N, L));
    //     vec3 diff = ubo.lights[i].color * albedo.rgb * NdotL * atten;

    //     // Specular part
    //     vec3 R = reflect(-L, N);
    //     float NdotR = max(0.0, dot(R, V));
    //     vec3 spec = ubo.lights[i].color * albedo.a * pow(NdotR, 8.0) * atten;

    //     result += diff + spec;    
    // }
    return result;
}

void main() 
{
    ivec2 attDim = textureSize(samplerPosition);
    ivec2 UV = ivec2(inUV * attDim);
    
    // // Debug display
    // if (ubo.debugDisplayTarget > 0) {
    //     switch (ubo.debugDisplayTarget) {
    //         case 1: 
    //             outFragcolor.rgb = texelFetch(samplerPosition, UV, 0).rgb;
    //             break;
    //         case 2: 
    //             outFragcolor.rgb = texelFetch(samplerNormal, UV, 0).rgb;
    //             break;
    //         case 3: 
    //             outFragcolor.rgb = texelFetch(samplerAlbedo, UV, 0).rgb;
    //             break;
    //         case 4: 
    //             outFragcolor.rgb = texelFetch(samplerAlbedo, UV, 0).aaa;
    //             break;
    //     }        
    //     outFragcolor.a = 1.0;
    //     return;
    // }

    #define ambient 0.15

    // Ambient part
    vec4 alb = resolve(samplerAlbedo, UV);
    vec3 fragColor = vec3(0.0);
    
    // Calualte lighting for every MSAA sample
    for (int i = 0; i < NUM_SAMPLES; i++)
    { 
        vec3 pos = texelFetch(samplerPosition, UV, i).rgb;
        vec3 normal = texelFetch(samplerNormal, UV, i).rgb;
        vec4 albedo = texelFetch(samplerAlbedo, UV, i);
        fragColor += calculateLighting(pos, normal, albedo);
    }

    fragColor = (alb.rgb * ambient) + fragColor / float(NUM_SAMPLES);
   
    outFragcolor = vec4(fragColor, 1.0);    
}