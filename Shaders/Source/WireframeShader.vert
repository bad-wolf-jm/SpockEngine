#version 450

layout( location = 0 ) in vec3 inPosition;

layout( push_constant ) uniform SceneUBO
{
    mat4 Model;
    mat4 View;
    mat4 Projection;
    vec4 Color;
}
Scene;

void main()
{
    gl_Position = Scene.Projection * Scene.View * Scene.Model * vec4( inPosition, 1.0 );
}
