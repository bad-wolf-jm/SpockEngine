#version 450

layout( push_constant ) uniform SceneUBO
{
    mat4 Model;
    mat4 View;
    mat4 Projection;
    vec4 Color;
}
Scene;

layout( location = 0 ) in vec3 fragPosition;
layout( location = 1 ) in vec3 fragNormal;

layout( location = 0 ) out vec4 outColor;

void main()
{
    vec3 L = normalize( inverse( Scene.View )[3].xyz - fragPosition );
    vec3 N = normalize( fragNormal );

    float NdotL = abs( dot( N, L ) );

    outColor = vec4( Scene.Color.rgb * NdotL, Scene.Color.a );
}