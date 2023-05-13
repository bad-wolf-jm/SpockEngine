#version 450

layout( push_constant ) uniform SceneUBO
{
    mat4 Model;
    mat4 View;
    mat4 Projection;
    vec4 Color;
}
Scene;

layout( location = 0 ) out vec4 o_Color;

void main() { o_Color = Scene.Color; }
