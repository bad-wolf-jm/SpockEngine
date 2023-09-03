#pragma once

#include "Core/Math/Types.h"
#include "Core/Types.h"
#include "Scene/VertexData.h"

namespace SE::Core::Primitives
{

    struct VertexBufferData
    {
        vector_t<SE::Core::VertexData> Vertices = {};
        vector_t<uint32_t>             Indices  = {};

        VertexBufferData()  = default;
        ~VertexBufferData() = default;

        void     PushFace( std::array<uint32_t, 3> l_Face );
        uint32_t PushVertex( const math::vec3 &position, const math::vec3 &normal, const math::vec2 texCoords );
    };

    struct WireframeVertexBufferData
    {
        vector_t<math::vec3> Vertices = {};
        vector_t<uint32_t>   Indices  = {};

        WireframeVertexBufferData()  = default;
        ~WireframeVertexBufferData() = default;

        void     PushEdge( std::array<uint32_t, 2> l_Face );
        uint32_t PushVertex( const math::vec3 &position );
    };

    VertexBufferData          CreateCube();
    WireframeVertexBufferData CreateCubeWireframe();

    VertexBufferData          CreatePlane( const math::ivec2 &subdivisions );
    WireframeVertexBufferData CreateWireframeGrid( const math::ivec2 &subdivisions );

    VertexBufferData          CreateSphere( const uint32_t rings, const uint32_t segments );
    WireframeVertexBufferData CreateWireframeSphere( const uint32_t rings, const uint32_t segments );

    VertexBufferData          CreateCylinder( const uint32_t rings, const uint32_t segments );
    WireframeVertexBufferData CreateWireframeCylinder( const uint32_t rings, const uint32_t segments );

    VertexBufferData          CreateCone( const uint32_t segments );
    WireframeVertexBufferData CreateWireframeCone( const uint32_t links, const uint32_t segments );

    VertexBufferData          CreateDisk( const uint32_t segments );
    WireframeVertexBufferData CreateCircle( const uint32_t segments );

    VertexBufferData          CreatePyramid( const uint32_t segments_x, const uint32_t segments_y );
    WireframeVertexBufferData CreateWireframePyramid( const uint32_t segments_x, const uint32_t segments_y );

} // namespace SE::Core::Primitives
