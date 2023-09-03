#include "Primitives.h"

namespace SE::Core::Primitives
{

    uint32_t VertexBufferData::PushVertex( const math::vec3 &position, const math::vec3 &normal, const math::vec2 texCoords )
    {
        SE::Core::VertexData l_NewVertex = {
            math::vec4( position, 1.0f ), normal, texCoords, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
        Vertices.emplace_back( l_NewVertex );
        return ( Vertices.size() - 1 );
    }

    void VertexBufferData::PushFace( std::array<uint32_t, 3> l_Face )
    {
        Indices.push_back( l_Face[0] );
        Indices.push_back( l_Face[1] );
        Indices.push_back( l_Face[2] );
    }

    void WireframeVertexBufferData::PushEdge( std::array<uint32_t, 2> l_Face )
    {
        Indices.push_back( l_Face[0] );
        Indices.push_back( l_Face[1] );
    }

    uint32_t WireframeVertexBufferData::PushVertex( const math::vec3 &position )
    {
        Vertices.push_back( math::vec4( position, 1.0f ) );
        return ( Vertices.size() - 1 );
    }

    VertexBufferData CreateCube()
    {
        // clang-format off
        VertexBufferData l_Cube;
        l_Cube.Vertices = {
            /* +Z */ { { -0.5f, -0.5f, 0.5f}, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, -0.5f, 0.5f}, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, 0.5f, 0.5f}, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { -0.5f, 0.5f, 0.5f}, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },

            /* -Z */ { { 0.5f, -0.5f, -0.5f}, { 0.0f, 0.0f, -1.0f }, { 0.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { -0.5f, -0.5f, -0.5f}, { 0.0f, 0.0f, -1.0f }, { 1.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { -0.5f, 0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, 0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },

            /* +Y */ { { -0.5f, 0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, 0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { -0.5f, 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },

            /* -Y */ { { -0.5f, -0.5f, -0.5f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, -0.5f, -0.5f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, -0.5f, 0.5f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { -0.5f, -0.5f, 0.5f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },

            /* +X */ { { 0.5f, -0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, 0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { 0.5f, 0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },

            /* -X */ { { -0.5f, -0.5f, -0.5f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { -0.5f, -0.5f, 0.5f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { -0.5f, 0.5f, 0.5f }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
                     { { -0.5f, 0.5f, -0.5f }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } } };

        l_Cube.Indices  = {
            /* +Z */ 0,  1,  2,  0,  2,  3,
            /* -Z */ 4,  5,  6,  4,  6,  7,
            /* +Y */ 8,  9,  10, 8,  10, 11,
            /* -Y */ 12, 13, 14, 12, 14, 15,
            /* +X */ 16, 17, 18, 16, 18, 19,
            /* -X */ 20, 21, 22, 20, 22, 23 };
        // clang-format on

        return l_Cube;
    }

    WireframeVertexBufferData CreateCubeWireframe()
    {
        WireframeVertexBufferData l_Cube;
        l_Cube.Vertices = { { -1.0f, -1.0f, 1.0f },  { 1.0f, -1.0f, 1.0f },  { 1.0f, 1.0f, 1.0f },  { -1.0f, 1.0f, 1.0f },
                            { -1.0f, -1.0f, -1.0f }, { 1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f }, { -1.0f, 1.0f, -1.0f } };
        l_Cube.Indices  = {
            0, 1, 1, 2, 2, 3, 3, 0, /* +Z */
            4, 5, 5, 6, 6, 7, 7, 4, /* -Z */
            1, 5, 2, 6,             /* +X */
            0, 4, 3, 7              /* -X */
        };
        return l_Cube;
    }

    VertexBufferData CreatePlane( const math::ivec2 &subdivisions )
    {
        const math::ivec2 vertexCount = subdivisions + math::ivec2{ 2 };
        const math::ivec2 faceCount   = subdivisions + math::ivec2{ 1 };

        VertexBufferData l_VertexData;
        l_VertexData.Vertices = vector_t<SE::Core::VertexData>( vertexCount.x * vertexCount.y );
        l_VertexData.Indices  = vector_t<uint32_t>( (uint32_t)faceCount.x * faceCount.y * 6 );

        /* Indices */
        // vector_t<uint32_t> indexData{(uint32_t) faceCount.x * faceCount.y * 6};
        std::size_t i = 0;
        for( int32_t y = 0; y != faceCount.y; ++y )
        {
            for( int32_t x = 0; x != faceCount.x; ++x )
            {
                /* 2--1 5
                    | / /|
                    |/ / |
                    0 3--4 */
                l_VertexData.Indices[i++] = uint32_t( y * vertexCount.x + x );
                l_VertexData.Indices[i++] = uint32_t( ( y + 1 ) * vertexCount.x + x + 1 );
                l_VertexData.Indices[i++] = uint32_t( ( y + 1 ) * vertexCount.x + x + 0 );
                l_VertexData.Indices[i++] = uint32_t( y * vertexCount.x + x );
                l_VertexData.Indices[i++] = uint32_t( y * vertexCount.x + x + 1 );
                l_VertexData.Indices[i++] = uint32_t( ( y + 1 ) * vertexCount.x + x + 1 );
            }
        }

        i = 0;
        for( int32_t y = 0; y != vertexCount.y; ++y )
        {
            for( int32_t x = 0; x != vertexCount.x; ++x )
            {
                l_VertexData.Vertices[i].Position =
                    math::vec3( ( x * 2.0f ) / faceCount.x - 1.0f, ( y * 2.0f ) / faceCount.y - 1.0f, 0.0 ) * 0.5f,
                l_VertexData.Vertices[i].Normal      = math::z_axis(),
                l_VertexData.Vertices[i].TexCoords_0 = { l_VertexData.Vertices[i].Position.x + 0.5f,
                                                         l_VertexData.Vertices[i].Position.y + 0.5f };
                i++;
            }
        }

        return l_VertexData;
    }

    WireframeVertexBufferData CreateWireframeGrid( const math::ivec2 &subdivisions )
    {
        const math::ivec2 vertexCount = subdivisions + math::ivec2{ 2 };
        const math::ivec2 faceCount   = subdivisions + math::ivec2{ 1 };

        WireframeVertexBufferData l_VertexData;
        l_VertexData.Vertices = vector_t<math::vec3>( vertexCount.x * vertexCount.y );
        l_VertexData.Indices =
            vector_t<uint32_t>( ( vertexCount.y * ( vertexCount.x - 1 ) * 2 + vertexCount.x * ( vertexCount.y - 1 ) * 2 ) );

        std::size_t i = 0;
        for( int32_t y = 0; y != vertexCount.y; ++y )
            for( int32_t x = 0; x != vertexCount.x; ++x )
                l_VertexData.Vertices[i++] =
                    math::vec4( ( x * 2.0f ) / faceCount.x - 1.0f, ( y * 2.0f ) / faceCount.y - 1.0f, 0.0, 1.0f );

        i = 0;
        for( int32_t y = 0; y != vertexCount.y; ++y )
        {
            for( int32_t x = 0; x != vertexCount.x; ++x )
            {
                /* 3    7
                    |    | ...
                    2    6
                    0--1 4--5 ... */
                if( x != vertexCount.x - 1 )
                {
                    l_VertexData.Indices[i++] = uint32_t( y * vertexCount.x + x );
                    l_VertexData.Indices[i++] = uint32_t( y * vertexCount.x + x + 1 );
                }

                if( y != vertexCount.y - 1 )
                {
                    l_VertexData.Indices[i++] = uint32_t( y * vertexCount.x + x );
                    l_VertexData.Indices[i++] = uint32_t( ( y + 1 ) * vertexCount.x + x );
                }
            }
        }

        return l_VertexData;
    }

    static constexpr float PI  = 3.141592654f;
    static constexpr float TAU = ( 2 * PI );

    static inline void NormalSpace( float a_Azimuth, float a_Elevation, math::vec3 &o_Normal )
    {
        float l_SinAzimuth   = math::sin( a_Azimuth );
        float l_CosAzimuth   = math::cos( a_Azimuth );
        float l_SinElevation = math::sin( a_Elevation );
        float l_CosElevation = math::cos( a_Elevation );
        o_Normal             = { l_CosElevation * l_SinAzimuth, l_SinElevation * l_SinAzimuth, l_CosAzimuth };
    }

    VertexBufferData CreateSphere( const uint32_t rings, const uint32_t segments )
    {
        VertexBufferData l_VertexData;
        float            l_DeltaElevation = PI / ( (float)rings );
        float            l_DeltaAzimuth   = ( 2 * PI ) / ( (float)segments );

        {
            // North pole vertex
            l_VertexData.PushVertex( { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.5f, 0.0f } );
            uint32_t l_PolarVertexIdx = l_VertexData.Vertices.size() - 1;

            // First ring: everything connects to the polar vertex
            float l_CurrentElevation = l_DeltaElevation;
            float l_CurrentAzimuth   = 0.0f;
            float l_RingRadius       = math::sin( l_CurrentElevation );
            float l_RingHeight       = math::cos( l_CurrentElevation );

            math::vec2 l_TexCoords = { 0.0f, ( (float)1.0f ) / ( (float)rings ) };
            l_VertexData.PushVertex(
                { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                l_TexCoords );
            uint32_t l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;

            for( uint32_t l_SegmentIdx = 1; l_SegmentIdx <= segments; l_SegmentIdx++ )
            {
                l_CurrentAzimuth += l_DeltaAzimuth;
                math::vec2 l_TexCoords = { ( (float)l_SegmentIdx ) / ( (float)segments ), ( (float)1.0f ) / ( (float)rings ) };

                l_VertexData.PushVertex(
                    { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                    { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                    l_TexCoords );

                uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
                l_VertexData.PushFace( { l_PolarVertexIdx, l_LastVertexIdx - 0, l_LastVertexIdx - 1 } );
            }
        }

        // Intermediate rings
        for( uint32_t l_RingIdx = 2; l_RingIdx < rings; l_RingIdx++ )
        {
            float      l_CurrentElevation = l_RingIdx * l_DeltaElevation;
            float      l_CurrentAzimuth   = 0.0f;
            float      l_RingRadius       = math::sin( l_CurrentElevation );
            float      l_RingHeight       = math::cos( l_CurrentElevation );
            math::vec2 l_TexCoords        = { 0.0f, ( (float)l_RingIdx ) / ( (float)rings ) };

            l_VertexData.PushVertex(
                { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                l_TexCoords );
            uint32_t l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;

            for( uint32_t l_SegmentIdx = 1; l_SegmentIdx < segments; l_SegmentIdx++ )
            {
                l_CurrentAzimuth += l_DeltaAzimuth;
                math::vec2 l_TexCoords = { ( (float)l_SegmentIdx ) / ( (float)segments ), ( (float)l_RingIdx ) / ( (float)rings ) };

                l_VertexData.PushVertex(
                    { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                    { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                    l_TexCoords );

                uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;

                l_VertexData.PushFace( { l_LastVertexIdx - 0, l_LastVertexIdx - 1, l_LastVertexIdx - 1 - segments } );
                l_VertexData.PushFace( { l_LastVertexIdx - 0, l_LastVertexIdx - 1 - segments, l_LastVertexIdx - segments } );
            }

            l_VertexData.PushFace( { l_FirstVertexIdx, (uint32_t)l_VertexData.Vertices.size() - 1,
                                     (uint32_t)l_VertexData.Vertices.size() - 1 - segments } );
            l_VertexData.PushFace(
                { l_FirstVertexIdx, (uint32_t)l_VertexData.Vertices.size() - 1 - segments, l_FirstVertexIdx - segments } );
        }

        // Last ring: everything connects to the polar vertex
        {
            // South pole vertex
            uint32_t   l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;
            math::vec3 l_Normal, l_Tangent, l_Bitangent;
            l_VertexData.PushVertex( { 0.0f, -1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f }, { 0.5f, 1.0f } );
            uint32_t l_PolarVertexIdx = l_VertexData.Vertices.size() - 1;

            for( uint32_t l_SegmentIdx = 0; l_SegmentIdx < segments; l_SegmentIdx++ )
                l_VertexData.PushFace(
                    { l_PolarVertexIdx,
                      ( l_SegmentIdx != segments - 1 ) ? ( l_PolarVertexIdx - l_SegmentIdx - 2 ) : l_PolarVertexIdx - 1,
                      l_PolarVertexIdx - l_SegmentIdx - 1 } );
        }

        return l_VertexData;
    }

    WireframeVertexBufferData CreateWireframeSphere( const uint32_t rings, const uint32_t segments )
    {
        WireframeVertexBufferData l_VertexData;

        return l_VertexData;
    }

    VertexBufferData CreateCylinder( const uint32_t rings, const uint32_t segments )
    {
        VertexBufferData l_VertexData;
        float            l_DeltaElevation = 2.0f / ( (float)rings );
        float            l_DeltaAzimuth   = ( 2 * PI ) / ( (float)segments );

        {
            // North pole vertex
            l_VertexData.PushVertex( { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.5f, 0.0f } );
            uint32_t l_PolarVertexIdx = l_VertexData.Vertices.size() - 1;

            // First ring: everything connects to the polar vertex
            float l_CurrentElevation = 0.0f;
            float l_CurrentAzimuth   = 0.0f;
            float l_RingRadius       = 1.0;
            float l_RingHeight       = 1.0f;

            math::vec3 l_Normal, l_Tangent, l_Bitangent;
            math::vec2 l_TexCoords = { 0.0f, ( (float)1.0f ) / ( (float)rings ) };
            NormalSpace( l_CurrentAzimuth, l_CurrentElevation, l_Normal );

            l_VertexData.PushVertex(
                { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) }, l_Normal,
                l_TexCoords );
            uint32_t l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;

            for( uint32_t l_SegmentIdx = 1; l_SegmentIdx <= segments; l_SegmentIdx++ )
            {
                l_CurrentAzimuth += l_DeltaAzimuth;
                math::vec3 l_Normal, l_Tangent, l_Bitangent;
                math::vec2 l_TexCoords = { ( (float)l_SegmentIdx ) / ( (float)segments ), ( (float)1.0f ) / ( (float)rings ) };
                NormalSpace( l_CurrentAzimuth, l_CurrentElevation, l_Normal );

                l_VertexData.PushVertex(
                    { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                    l_Normal, l_TexCoords );

                uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
                l_VertexData.PushFace( { l_PolarVertexIdx, l_LastVertexIdx - 0, l_LastVertexIdx - 1 } );
            }
        }

        // Intermediate rings
        for( uint32_t l_RingIdx = 0; l_RingIdx <= rings; l_RingIdx++ )
        {
            float l_CurrentElevation = 1.0f - l_RingIdx * l_DeltaElevation;
            float l_CurrentAzimuth   = 0.0f;
            float l_RingRadius       = 1.0;
            float l_RingHeight       = l_CurrentElevation;

            math::vec3 l_Normal, l_Tangent, l_Bitangent;
            math::vec2 l_TexCoords = { 0.0f, ( (float)l_RingIdx ) / ( (float)rings ) };
            NormalSpace( l_CurrentAzimuth, l_CurrentElevation, l_Normal );

            l_VertexData.PushVertex(
                { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) }, l_Normal,
                l_TexCoords );
            uint32_t l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;

            for( uint32_t l_SegmentIdx = 1; l_SegmentIdx < segments; l_SegmentIdx++ )
            {
                l_CurrentAzimuth += l_DeltaAzimuth;
                math::vec2 l_TexCoords = { ( (float)l_SegmentIdx ) / ( (float)segments ), ( (float)l_RingIdx ) / ( (float)rings ) };
                math::vec3 l_Normal, l_Tangent, l_Bitangent;
                NormalSpace( l_CurrentAzimuth, l_CurrentElevation, l_Normal );

                l_VertexData.PushVertex(
                    { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                    l_Normal, l_TexCoords );

                uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;

                l_VertexData.PushFace( { l_LastVertexIdx - 0, l_LastVertexIdx - 1, l_LastVertexIdx - 1 - segments } );
                l_VertexData.PushFace( { l_LastVertexIdx - 0, l_LastVertexIdx - 1 - segments, l_LastVertexIdx - segments } );
            }

            l_VertexData.PushFace( { (uint32_t)l_VertexData.Vertices.size() - 1, l_FirstVertexIdx - segments, l_FirstVertexIdx } );
            l_VertexData.PushFace( { (uint32_t)l_VertexData.Vertices.size() - 1, (uint32_t)l_VertexData.Vertices.size() - 1 - segments,
                                     l_FirstVertexIdx - segments } );
        }

        // Last ring: everything connects to the polar vertex
        {
            // South pole vertex
            uint32_t   l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;
            math::vec3 l_Normal, l_Tangent, l_Bitangent;
            l_VertexData.PushVertex( { 0.0f, -1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f }, { 0.5f, 1.0f } );
            uint32_t l_PolarVertexIdx = l_VertexData.Vertices.size() - 1;

            for( uint32_t l_SegmentIdx = 0; l_SegmentIdx < segments; l_SegmentIdx++ )
                l_VertexData.PushFace(
                    { l_PolarVertexIdx,
                      ( l_SegmentIdx != segments - 1 ) ? ( l_PolarVertexIdx - l_SegmentIdx - 2 ) : l_PolarVertexIdx - 1,
                      l_PolarVertexIdx - l_SegmentIdx - 1 } );
        }

        return l_VertexData;
    }

    WireframeVertexBufferData CreateWireframeCylinder( const uint32_t rings, const uint32_t segments )
    {
        WireframeVertexBufferData l_VertexData;

        return l_VertexData;
    }

    VertexBufferData CreateCone( const uint32_t segments )
    {
        VertexBufferData l_VertexData;
        float            l_DeltaAzimuth = ( 2 * PI ) / ( (float)segments );

        {
            // North pole vertex
            l_VertexData.PushVertex( { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.5f, 0.0f } );
            uint32_t l_PolarVertexIdx = l_VertexData.Vertices.size() - 1;

            // First ring: everything connects to the polar vertex
            float l_CurrentElevation = 0.0f;
            float l_CurrentAzimuth   = 0.0f;
            float l_RingRadius       = 1.0;
            float l_RingHeight       = 0.0f;

            math::vec3 l_Normal, l_Tangent, l_Bitangent;
            math::vec2 l_TexCoords = { 0.0f, ( (float)1.0f ) / ( (float)1.0f ) };
            NormalSpace( l_CurrentAzimuth, l_CurrentElevation, l_Normal );

            l_VertexData.PushVertex(
                { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) }, l_Normal,
                l_TexCoords );
            uint32_t l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;

            for( uint32_t l_SegmentIdx = 1; l_SegmentIdx <= segments; l_SegmentIdx++ )
            {
                l_CurrentAzimuth += l_DeltaAzimuth;
                math::vec3 l_Normal, l_Tangent, l_Bitangent;
                math::vec2 l_TexCoords = { ( (float)l_SegmentIdx ) / ( (float)segments ), ( (float)1.0f ) / ( (float)1.0f ) };
                NormalSpace( l_CurrentAzimuth, l_CurrentElevation, l_Normal );

                l_VertexData.PushVertex(
                    { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                    l_Normal, l_TexCoords );

                uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
                l_VertexData.PushFace( { l_PolarVertexIdx, l_LastVertexIdx - 0, l_LastVertexIdx - 1 } );
            }
        }

        // Last ring: everything connects to the polar vertex
        {
            // South pole vertex
            uint32_t   l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;
            math::vec3 l_Normal, l_Tangent, l_Bitangent;
            l_VertexData.PushVertex( { 0.0f, 0.0f, 0.0f }, { 0.0f, -1.0f, 0.0f }, { 0.5f, 1.0f } );
            uint32_t l_PolarVertexIdx = l_VertexData.Vertices.size() - 1;

            for( uint32_t l_SegmentIdx = 0; l_SegmentIdx < segments; l_SegmentIdx++ )
                l_VertexData.PushFace(
                    { l_PolarVertexIdx,
                      ( l_SegmentIdx != segments - 1 ) ? ( l_PolarVertexIdx - l_SegmentIdx - 2 ) : l_PolarVertexIdx - 1,
                      l_PolarVertexIdx - l_SegmentIdx - 1 } );
        }

        return l_VertexData;
    }

    WireframeVertexBufferData CreateWireframeCone( const uint32_t links, const uint32_t segments )
    {
        WireframeVertexBufferData l_VertexData;
        float                     l_DeltaAzimuth = ( 2 * PI ) / ( (float)segments );

        // North pole vertex
        l_VertexData.PushVertex( { 0.0f, 0.0f, 1.0f } );
        uint32_t l_PolarVertexIdx = l_VertexData.Vertices.size() - 1;

        // First ring: everything connects to the polar vertex
        float l_CurrentAzimuth = 0.0f;
        float l_RingRadius     = 1.0;

        l_VertexData.PushVertex(
            { l_RingRadius * math::cos( l_CurrentAzimuth ), 0.0f, l_RingRadius * math::sin( l_CurrentAzimuth ) } );
        uint32_t l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;

        for( uint32_t l_SegmentIdx = 1; l_SegmentIdx <= segments; l_SegmentIdx++ )
        {
            l_CurrentAzimuth += l_DeltaAzimuth;

            l_VertexData.PushVertex(
                { l_RingRadius * math::cos( l_CurrentAzimuth ), 0.0f, l_RingRadius * math::sin( l_CurrentAzimuth ) } );

            uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
            l_VertexData.PushEdge( { l_LastVertexIdx, l_LastVertexIdx - 1 } );
        }

        l_VertexData.PushVertex( { 0.0f, 1.0f, 0.0f } );
        uint32_t l_PeakIdx = l_VertexData.Vertices.size() - 1;

        l_CurrentAzimuth = 0.0f;
        l_DeltaAzimuth   = ( 2 * PI ) / ( (float)links );
        for( uint32_t l_LinkIdx = 1; l_LinkIdx <= links; l_LinkIdx++ )
        {
            l_CurrentAzimuth += l_DeltaAzimuth;

            l_VertexData.PushVertex(
                { l_RingRadius * math::cos( l_CurrentAzimuth ), 0.0f, l_RingRadius * math::sin( l_CurrentAzimuth ) } );

            uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
            l_VertexData.PushEdge( { l_LastVertexIdx, l_PeakIdx } );
        }

        return l_VertexData;
    }

    VertexBufferData CreateDisk( const uint32_t segments )
    {
        VertexBufferData l_VertexData;
        float            l_DeltaAzimuth = ( 2 * PI ) / ( (float)segments );

        {
            // North pole vertex
            l_VertexData.PushVertex( { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.5f, 0.0f } );
            uint32_t l_PolarVertexIdx = l_VertexData.Vertices.size() - 1;

            // First ring: everything connects to the polar vertex
            float l_CurrentElevation = 0.0f;
            float l_CurrentAzimuth   = 0.0f;
            float l_RingRadius       = 1.0;
            float l_RingHeight       = 1.0f;

            math::vec3 l_Normal, l_Tangent, l_Bitangent;
            math::vec2 l_TexCoords = { 0.0f, 1.0f };
            NormalSpace( l_CurrentAzimuth, l_CurrentElevation, l_Normal );

            l_VertexData.PushVertex(
                { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) }, l_Normal,
                l_TexCoords );
            uint32_t l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;

            for( uint32_t l_SegmentIdx = 1; l_SegmentIdx <= segments; l_SegmentIdx++ )
            {
                l_CurrentAzimuth += l_DeltaAzimuth;
                math::vec3 l_Normal, l_Tangent, l_Bitangent;
                math::vec2 l_TexCoords = { ( (float)l_SegmentIdx ) / ( (float)segments ), 1.0f };
                NormalSpace( l_CurrentAzimuth, l_CurrentElevation, l_Normal );

                l_VertexData.PushVertex(
                    { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingHeight, l_RingRadius * math::sin( l_CurrentAzimuth ) },
                    l_Normal, l_TexCoords );

                uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
                l_VertexData.PushFace( { l_PolarVertexIdx, l_LastVertexIdx - 0, l_LastVertexIdx - 1 } );
            }
        }

        return l_VertexData;
    }

    WireframeVertexBufferData CreateCircle( const uint32_t segments )
    {
        WireframeVertexBufferData l_VertexData;

        float l_DeltaAzimuth = ( 2 * PI ) / ( (float)segments );

        // North pole vertex
        l_VertexData.PushVertex( { 0.0f, 1.0f, 0.0f } );
        uint32_t l_PolarVertexIdx = l_VertexData.Vertices.size() - 1;

        // First ring: everything connects to the polar vertex
        float l_CurrentAzimuth = 0.0f;
        float l_RingRadius     = 1.0;

        l_VertexData.PushVertex(
            { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingRadius * math::sin( l_CurrentAzimuth ), 0.0f } );
        uint32_t l_FirstVertexIdx = l_VertexData.Vertices.size() - 1;

        for( uint32_t l_SegmentIdx = 1; l_SegmentIdx <= segments; l_SegmentIdx++ )
        {
            l_CurrentAzimuth += l_DeltaAzimuth;

            l_VertexData.PushVertex(
                { l_RingRadius * math::cos( l_CurrentAzimuth ), l_RingRadius * math::sin( l_CurrentAzimuth ), 0.0f } );

            uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
            l_VertexData.PushEdge( { l_LastVertexIdx, l_LastVertexIdx - 1 } );
        }

        return l_VertexData;
    }

    VertexBufferData CreatePyramid( const uint32_t segments_x, const uint32_t segments_y )
    {
        VertexBufferData l_VertexData;

        return l_VertexData;
    }

    WireframeVertexBufferData CreateWireframePyramid( const uint32_t segments_x, const uint32_t segments_y )
    {
        WireframeVertexBufferData l_VertexData;

        // North pole vertex
        l_VertexData.PushVertex( { 0.0f, 1.0f, 0.0f } );
        l_VertexData.PushVertex( { -0.5f, 0.0f, -0.5f } );
        l_VertexData.PushVertex( { -0.5f, 0.0f, 0.5f } );
        l_VertexData.PushVertex( { 0.5f, 0.0f, -0.5f } );
        l_VertexData.PushVertex( { 0.5f, 0.0f, 0.5f } );

        l_VertexData.PushEdge( { 0, 1 } );
        l_VertexData.PushEdge( { 0, 2 } );
        l_VertexData.PushEdge( { 0, 3 } );
        l_VertexData.PushEdge( { 0, 4 } );

        for( uint32_t l_SegmentIdx = 0; l_SegmentIdx <= segments_y; l_SegmentIdx++ )
        {
            float y_value = -0.5f + static_cast<float>( l_SegmentIdx ) / static_cast<float>( segments_y );
            l_VertexData.PushVertex( { -0.5f, 0.0f, y_value } );
            uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
            if( l_SegmentIdx > 0 )
                l_VertexData.PushEdge( { l_LastVertexIdx, l_LastVertexIdx - 1 } );
        }

        for( uint32_t l_SegmentIdx = 0; l_SegmentIdx <= segments_y; l_SegmentIdx++ )
        {
            float y_value = -0.5f + static_cast<float>( l_SegmentIdx ) / static_cast<float>( segments_y );
            l_VertexData.PushVertex( { 0.5f, 0.0f, y_value } );
            uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
            if( l_SegmentIdx > 0 )
                l_VertexData.PushEdge( { l_LastVertexIdx, l_LastVertexIdx - 1 } );
        }

        for( uint32_t l_SegmentIdx = 0; l_SegmentIdx <= segments_x; l_SegmentIdx++ )
        {
            float y_value = -0.5f + static_cast<float>( l_SegmentIdx ) / static_cast<float>( segments_y );
            l_VertexData.PushVertex( { y_value, 0.0f, -0.5 } );
            uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
            if( l_SegmentIdx > 0 )
                l_VertexData.PushEdge( { l_LastVertexIdx, l_LastVertexIdx - 1 } );
        }

        for( uint32_t l_SegmentIdx = 0; l_SegmentIdx <= segments_x; l_SegmentIdx++ )
        {
            float y_value = -0.5f + static_cast<float>( l_SegmentIdx ) / static_cast<float>( segments_y );
            l_VertexData.PushVertex( { y_value, 0.0f, 0.5 } );
            uint32_t l_LastVertexIdx = l_VertexData.Vertices.size() - 1;
            if( l_SegmentIdx > 0 )
                l_VertexData.PushEdge( { l_LastVertexIdx, l_LastVertexIdx - 1 } );
        }

        return l_VertexData;
    }

} // namespace SE::Core::Primitives
