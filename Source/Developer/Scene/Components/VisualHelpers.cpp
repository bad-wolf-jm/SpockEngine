#include "VisualHelpers.h"

#include "Core/Math/Types.h"

#include "Developer/Scene/Primitives/Arrow.h"
#include "Developer/Scene/Primitives/Primitives.h"

namespace LTSE::Core::EntityComponentSystem::Components
{

    static std::vector<SimpleVertexData> ToSimpleVertex( std::vector<VertexData> a_Source, math::mat4 a_Transform )
    {
        std::vector<SimpleVertexData> l_Result( a_Source.size() );
        for( uint32_t i = 0; i < a_Source.size(); i++ )
        {
            l_Result[i].Position = a_Transform * a_Source[i].Position;
            l_Result[i].Normal   = a_Source[i].Normal;
        }

        return l_Result;
    }

    static void ApplyTransform( std::vector<VertexData> &a_Source, math::mat4 a_Transform )
    {
        for( uint32_t i = 0; i < a_Source.size(); i++ )
        {
            a_Source[i].Position = a_Transform * a_Source[i].Position;
        }
    }

    static void ApplyTransform( std::vector<math::vec4> &a_Source, math::mat4 a_Transform )
    {
        for( uint32_t i = 0; i < a_Source.size(); i++ )
        {
            a_Source[i] = a_Transform * a_Source[i];
        }
    }

    void CubeMeshData::UpdateMesh( GraphicContext &a_GraphicContext )
    {
        math::mat4 l_ScalingTransform                   = math::Scale( math::mat4( 1.0f ), math::vec3{ SideLength, SideLength, SideLength } );
        LTSE::Core::Primitives::VertexBufferData l_Cube = LTSE::Core::Primitives::CreateCube();
        std::vector<SimpleVertexData> l_CubeVertexData  = ToSimpleVertex( l_Cube.Vertices, l_ScalingTransform );

        Mesh.Vertices = New<Buffer>( a_GraphicContext, l_CubeVertexData, eBufferBindType::VERTEX_BUFFER, false, false, true, true );
        Mesh.Indices  = New<Buffer>( a_GraphicContext, l_Cube.Indices, eBufferBindType::INDEX_BUFFER, false, false, true, true );
    }

    void ConeMeshData::UpdateMesh( GraphicContext &a_GraphicContext )
    {
        LTSE::Core::Primitives::WireframeVertexBufferData l_Cone = LTSE::Core::Primitives::CreateWireframeCone( Divisions, Segments );

        Mesh.Vertices = New<Buffer>( a_GraphicContext, l_Cone.Vertices, eBufferBindType::VERTEX_BUFFER, false, false, true, true );
        Mesh.Indices  = New<Buffer>( a_GraphicContext, l_Cone.Indices, eBufferBindType::INDEX_BUFFER, false, false, true, true );
    }

    void ArrowMeshData::UpdateMesh( GraphicContext &a_GraphicContext )
    {
        LTSE::Core::Primitives::VertexBufferData l_Shaft = LTSE::Core::Primitives::CreateCylinder( 3, Segments );
        math::mat4 l_ShaftTransform = math::Translation( math::vec3{ 0.0f, 0.46f, 0.0f } ) * math::Scale( math::mat4( 1.0f ), math::vec3{ .0075f, 0.5 * 0.95f, .0075f } );
        ApplyTransform( l_Shaft.Vertices, l_ShaftTransform );

        LTSE::Core::Primitives::VertexBufferData l_Tip = LTSE::Core::Primitives::CreateCone( Segments );
        math::mat4 l_TipTransform                      = math::Translation( math::vec3{ .0f, .90f, 0.0f } ) * math::Scale( math::mat4( 1.0f ), math::vec3{ .025f, .1f, .025f } );
        ApplyTransform( l_Tip.Vertices, l_TipTransform );

        uint32_t l_ShaftOffset = l_Shaft.Vertices.size();
        for( uint32_t i = 0; i < l_Tip.Indices.size(); i++ )
        {
            l_Tip.Indices[i] += l_ShaftOffset;
        }

        LTSE::Core::Primitives::VertexBufferData l_ArrowMeshData{};
        l_ArrowMeshData.Vertices.insert( l_ArrowMeshData.Vertices.end(), l_Shaft.Vertices.begin(), l_Shaft.Vertices.end() );
        l_ArrowMeshData.Vertices.insert( l_ArrowMeshData.Vertices.end(), l_Tip.Vertices.begin(), l_Tip.Vertices.end() );
        l_ArrowMeshData.Indices.insert( l_ArrowMeshData.Indices.end(), l_Shaft.Indices.begin(), l_Shaft.Indices.end() );
        l_ArrowMeshData.Indices.insert( l_ArrowMeshData.Indices.end(), l_Tip.Indices.begin(), l_Tip.Indices.end() );

        std::vector<SimpleVertexData> l_ArrowVertexData = ToSimpleVertex( l_ArrowMeshData.Vertices, math::mat4( 1.0f ) );

        Mesh.Vertices = New<Buffer>( a_GraphicContext, l_ArrowVertexData, eBufferBindType::VERTEX_BUFFER, false, false, true, true );
        Mesh.Indices  = New<Buffer>( a_GraphicContext, l_ArrowMeshData.Indices, eBufferBindType::INDEX_BUFFER, false, false, true, true );
    }

    void CircleMeshData::UpdateMesh( GraphicContext &a_GraphicContext )
    {
        LTSE::Core::Primitives::WireframeVertexBufferData l_Circle = LTSE::Core::Primitives::CreateCircle( Segments );

        math::mat4 l_RadiusTransform = math::Scale( math::mat4( 1.0f ), math::vec3{ Radius, Radius, Radius } );
        ApplyTransform( l_Circle.Vertices, l_RadiusTransform );

        Mesh.Vertices = New<Buffer>( a_GraphicContext, l_Circle.Vertices, eBufferBindType::VERTEX_BUFFER, false, false, true, true );
        Mesh.Indices  = New<Buffer>( a_GraphicContext, l_Circle.Indices, eBufferBindType::INDEX_BUFFER, false, false, true, true );
    }

    void PyramidMeshData::UpdateMesh( GraphicContext &a_GraphicContext )
    {
        LTSE::Core::Primitives::WireframeVertexBufferData l_Pyramid = LTSE::Core::Primitives::CreateWireframePyramid( 64, 64 );

        Mesh.Vertices = New<Buffer>( a_GraphicContext, l_Pyramid.Vertices, eBufferBindType::VERTEX_BUFFER, false, false, true, true );
        Mesh.Indices  = New<Buffer>( a_GraphicContext, l_Pyramid.Indices, eBufferBindType::INDEX_BUFFER, false, false, true, true );
    }

    void SurfaceMeshData::UpdateMesh( GraphicContext &a_GraphicContext )
    {
        LTSE::Core::Primitives::VertexBufferData l_Plane = LTSE::Core::Primitives::CreatePlane( math::ivec2{ 32, 32 } );
        std::vector<SimpleVertexData> l_PlaneVertexData  = ToSimpleVertex( l_Plane.Vertices, math::mat4( 1.0f ) ); //(l_Plane.Vertices.size());

        Mesh.Vertices = New<Buffer>( a_GraphicContext, l_PlaneVertexData, eBufferBindType::VERTEX_BUFFER, false, false, true, true );
        Mesh.Indices  = New<Buffer>( a_GraphicContext, l_Plane.Indices, eBufferBindType::INDEX_BUFFER, false, false, true, true );
    }

    void SpotlightHelperComponent::UpdateMesh( GraphicContext &a_GraphicContext )
    {
        Origin.UpdateMesh( a_GraphicContext );

        LTSE::Core::Primitives::WireframeVertexBufferData l_Spot = LTSE::Core::Primitives::CreateWireframeCone( Spot.Divisions, Spot.Segments );
        math::mat4 l_Transform                                   = math::Translation( math::vec3{ 0.0f, 0.0f, -1.0f } ) * math::Rotation( 90.0_degf, math::x_axis() );
        ApplyTransform( l_Spot.Vertices, l_Transform );

        Spot.Mesh.Vertices = New<Buffer>( a_GraphicContext, l_Spot.Vertices, eBufferBindType::VERTEX_BUFFER, false, false, true, true );
        Spot.Mesh.Indices  = New<Buffer>( a_GraphicContext, l_Spot.Indices, eBufferBindType::INDEX_BUFFER, false, false, true, true );
    }

    void FieldOfViewHelperComponent::UpdateMesh( GraphicContext &a_GraphicContext )
    {
        LTSE::Core::Primitives::WireframeVertexBufferData l_Pyramid = LTSE::Core::Primitives::CreateWireframePyramid( 64, 64 );

        math::mat4 l_Transform = math::Translation( math::vec3{ 0.0f, 0.0f, -1.0f } ) * math::Rotation( 90.0_degf, math::x_axis() );
        ApplyTransform( l_Pyramid.Vertices, l_Transform );

        Outline.Mesh.Vertices = New<Buffer>( a_GraphicContext, l_Pyramid.Vertices, eBufferBindType::VERTEX_BUFFER, false, false, true, true );
        Outline.Mesh.Indices  = New<Buffer>( a_GraphicContext, l_Pyramid.Indices, eBufferBindType::INDEX_BUFFER, false, false, true, true );

        LTSE::Core::Primitives::VertexBufferData l_Plane = LTSE::Core::Primitives::CreatePlane( math::ivec2{ 32, 32 } );
        std::vector<SimpleVertexData> l_PlaneVertexData( l_Plane.Vertices.size() );
        for( uint32_t i = 0; i < l_Plane.Vertices.size(); i++ )
        {
            l_PlaneVertexData[i].Position = math::vec4( math::vec2( l_Plane.Vertices[i].Position ), -1.0f, 1.0f );
            l_PlaneVertexData[i].Normal   = l_Plane.Vertices[i].Normal;
        }

        OuterLimit.Mesh.Vertices = New<Buffer>( a_GraphicContext, l_PlaneVertexData, eBufferBindType::VERTEX_BUFFER, false, false, true, true );
        OuterLimit.Mesh.Indices  = New<Buffer>( a_GraphicContext, l_Plane.Indices, eBufferBindType::INDEX_BUFFER, false, false, true, true );
    }

} // namespace LTSE::Core::EntityComponentSystem::Components