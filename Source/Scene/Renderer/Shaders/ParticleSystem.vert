#version 450

// Input vertex data, different for all executions of this shader.
layout( location = 0 ) in vec3 squareVertices;

// Instanced
layout( location = 1 ) in vec4 xyzs;  // Position of the center of the particle and size of the square
layout( location = 2 ) in vec4 color; // Particle color

// Output data ; will be interpolated for each fragment.
layout( location = 0 ) out vec2 UV;
layout( location = 1 ) out vec4 particlecolor;

layout( set = 0, binding = 0 ) uniform UBO
{
    mat4 Model;
    mat4 View;
    mat4 Projection;
    float ParticleSize;
}
ubo;

void main()
{
    float particleSize  = ubo.ParticleSize > 0.0f ? ubo.ParticleSize : xyzs.w; // because we encoded it this way.
    vec3 particleCenter = xyzs.xyz;

    vec4 l_CameraRight = transpose(ubo.View) * vec4( 1.0f, 0.0f, 0.0f, 1.0f );
    vec4 l_CameraUp    = transpose(ubo.View) * vec4( 0.0f, 1.0f, 0.0f, 1.0f );

    vec3 vertexPosition_worldspace = ( ubo.Model * vec4( particleCenter, 1.0f ) ).xyz + normalize( l_CameraRight.xyz ) * squareVertices.x * particleSize +
                                     normalize( l_CameraUp.xyz ) * squareVertices.y * particleSize;

    // Output position of the vertex
    gl_Position   = ubo.Projection * ubo.View * vec4( vertexPosition_worldspace, 1.0f );

    // UV of the vertex. No special space for this one.
    UV            = squareVertices.xy + vec2( 0.5, 0.5 );
    particlecolor = color;
}