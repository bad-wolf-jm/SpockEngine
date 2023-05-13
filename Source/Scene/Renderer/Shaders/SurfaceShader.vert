#version 450

layout( location = 0 ) in vec3 position;

layout( set = 0, binding = 0 ) uniform UBO
{
    mat4 ModelMatrix;
    mat4 ViewMatrix;
    mat4 ProjectionMatrix;
    vec4 SubresourceRectangle;
}
ubo;

layout( set = 0, binding = 1 ) uniform sampler2D a_HeightMap;

void main()
{
    vec4 l_TransformedPosition = ubo.ModelMatrix * vec4( position.x, position.y, texture( a_HeightMap, position.xy ).r, 1.0 );
    gl_Position                = ubo.ProjectionMatrix * ubo.ViewMatrix * l_TransformedPosition;
}
