#version 450

layout( set = 0, binding = 0 ) uniform ViewUniforms
{
    mat4 view;
    mat4 proj;
}
view;

layout( location = 0 ) out float near; // 0.01
layout( location = 1 ) out float far;  // 100
layout( location = 2 ) out vec3 nearPoint;
layout( location = 3 ) out vec3 farPoint;
layout( location = 4 ) out mat4 fragView;
layout( location = 8 ) out mat4 fragProj;

// Grid position are in clipped space
vec4 gridPlane[6] = vec4[]( vec4( 1, 1, 0, 1 ), vec4( -1, -1, 0, 1 ), vec4( -1, 1, 0, 1 ), vec4( -1, -1, 0, 1 ), vec4( 1, 1, 0, 1 ), vec4( 1, -1, 0, 1 ) );

vec3 UnprojectPoint( float x, float y, float z, mat4 view, mat4 projection )
{
    mat4 viewInv          = inverse( view );
    mat4 projInv          = inverse( projection );
    vec4 unprojectedPoint = viewInv * projInv * vec4( x, y, z, 1.0 );
    return unprojectedPoint.xyz / unprojectedPoint.w;
}

void main()
{
    vec3 p      = gridPlane[gl_VertexIndex].xyz;
    near        = 0.01;
    far         = 100.0f;
    fragView    = view.view;
    fragProj    = view.proj;
    nearPoint   = UnprojectPoint( p.x, p.y, 0.0, view.view, view.proj ).xyz; // unprojecting on the near plane
    farPoint    = UnprojectPoint( p.x, p.y, 1.0, view.view, view.proj ).xyz; // unprojecting on the far plane
    gl_Position = vec4( p, 1.0 );                                            // using directly the clipped coordinates
}