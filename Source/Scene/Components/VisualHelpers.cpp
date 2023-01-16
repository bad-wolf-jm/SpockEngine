#include "VisualHelpers.h"

#include "Core/Math/Types.h"

#include "Scene/Primitives/Arrow.h"
#include "Scene/Primitives/Primitives.h"

namespace SE::Core::EntityComponentSystem::Components
{

    static std::vector<SimpleVertexData> ToSimpleVertex( std::vector<VertexData> a_Source, math::mat4 a_Transform )
    {
        std::vector<SimpleVertexData> l_Result( a_Source.size() );
        for( uint32_t i = 0; i < a_Source.size(); i++ )
        {
            l_Result[i].Position = math::vec3( a_Transform * math::vec4( a_Source[i].Position, 1.0f ) );
            l_Result[i].Normal   = a_Source[i].Normal;
        }

        return l_Result;
    }

    static void ApplyTransform( std::vector<VertexData> &a_Source, math::mat4 a_Transform )
    {
        for( uint32_t i = 0; i < a_Source.size(); i++ )
        {
            a_Source[i].Position = math::vec3( a_Transform * math::vec4( a_Source[i].Position, 1.0f ) );
        }
    }

    static void ApplyTransform( std::vector<math::vec3> &a_Source, math::mat4 a_Transform )
    {
        for( uint32_t i = 0; i < a_Source.size(); i++ )
        {
            a_Source[i] = math::vec3( a_Transform * math::vec4( a_Source[i], 1.0f ) );
        }
    }

    void CubeMeshData::UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
    {
        math::mat4 lScalingTransform = math::Scale( math::mat4( 1.0f ), math::vec3{ SideLength, SideLength, SideLength } );
        SE::Core::Primitives::VertexBufferData lCube           = SE::Core::Primitives::CreateCube();
        std::vector<SimpleVertexData>          lCubeVertexData = ToSimpleVertex( lCube.Vertices, lScalingTransform );

        Mesh.Vertices = New<VkGpuBuffer>( a_GraphicContext, lCubeVertexData, eBufferType::VERTEX_BUFFER, false, true, true, true );
        Mesh.Indices  = New<VkGpuBuffer>( a_GraphicContext, lCube.Indices, eBufferType::INDEX_BUFFER, false, true, true, true );
    }

    void ConeMeshData::UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
    {
        SE::Core::Primitives::WireframeVertexBufferData l_Cone = SE::Core::Primitives::CreateWireframeCone( Divisions, Segments );

        Mesh.Vertices = New<VkGpuBuffer>( a_GraphicContext, l_Cone.Vertices, eBufferType::VERTEX_BUFFER, false, true, true, true );
        Mesh.Indices  = New<VkGpuBuffer>( a_GraphicContext, l_Cone.Indices, eBufferType::INDEX_BUFFER, false, true, true, true );
    }

    void ArrowMeshData::UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
    {
        SE::Core::Primitives::VertexBufferData l_Shaft          = SE::Core::Primitives::CreateCylinder( 3, Segments );
        math::mat4                             l_ShaftTransform = math::Translation( math::vec3{ 0.0f, 0.46f, 0.0f } ) *
                                      math::Scale( math::mat4( 1.0f ), math::vec3{ .0075f, 0.5 * 0.95f, .0075f } );
        ApplyTransform( l_Shaft.Vertices, l_ShaftTransform );

        SE::Core::Primitives::VertexBufferData l_Tip = SE::Core::Primitives::CreateCone( Segments );
        math::mat4                             l_TipTransform =
            math::Translation( math::vec3{ .0f, .90f, 0.0f } ) * math::Scale( math::mat4( 1.0f ), math::vec3{ .025f, .1f, .025f } );
        ApplyTransform( l_Tip.Vertices, l_TipTransform );

        uint32_t l_ShaftOffset = l_Shaft.Vertices.size();
        for( uint32_t i = 0; i < l_Tip.Indices.size(); i++ )
        {
            l_Tip.Indices[i] += l_ShaftOffset;
        }

        SE::Core::Primitives::VertexBufferData l_ArrowMeshData{};
        l_ArrowMeshData.Vertices.insert( l_ArrowMeshData.Vertices.end(), l_Shaft.Vertices.begin(), l_Shaft.Vertices.end() );
        l_ArrowMeshData.Vertices.insert( l_ArrowMeshData.Vertices.end(), l_Tip.Vertices.begin(), l_Tip.Vertices.end() );
        l_ArrowMeshData.Indices.insert( l_ArrowMeshData.Indices.end(), l_Shaft.Indices.begin(), l_Shaft.Indices.end() );
        l_ArrowMeshData.Indices.insert( l_ArrowMeshData.Indices.end(), l_Tip.Indices.begin(), l_Tip.Indices.end() );

        std::vector<SimpleVertexData> l_ArrowVertexData = ToSimpleVertex( l_ArrowMeshData.Vertices, math::mat4( 1.0f ) );

        Mesh.Vertices = New<VkGpuBuffer>( a_GraphicContext, l_ArrowVertexData, eBufferType::VERTEX_BUFFER, false, true, true, true );
        Mesh.Indices =
            New<VkGpuBuffer>( a_GraphicContext, l_ArrowMeshData.Indices, eBufferType::INDEX_BUFFER, false, true, true, true );
    }

    void CircleMeshData::UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
    {
        SE::Core::Primitives::WireframeVertexBufferData l_Circle = SE::Core::Primitives::CreateCircle( Segments );

        math::mat4 l_RadiusTransform = math::Scale( math::mat4( 1.0f ), math::vec3{ Radius, Radius, Radius } );
        ApplyTransform( l_Circle.Vertices, l_RadiusTransform );

        Mesh.Vertices = New<VkGpuBuffer>( a_GraphicContext, l_Circle.Vertices, eBufferType::VERTEX_BUFFER, false, true, true, true );
        Mesh.Indices  = New<VkGpuBuffer>( a_GraphicContext, l_Circle.Indices, eBufferType::INDEX_BUFFER, false, true, true, true );
    }

    void PyramidMeshData::UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
    {
        SE::Core::Primitives::WireframeVertexBufferData l_Pyramid = SE::Core::Primitives::CreateWireframePyramid( 64, 64 );

        Mesh.Vertices = New<VkGpuBuffer>( a_GraphicContext, l_Pyramid.Vertices, eBufferType::VERTEX_BUFFER, false, true, true, true );
        Mesh.Indices  = New<VkGpuBuffer>( a_GraphicContext, l_Pyramid.Indices, eBufferType::INDEX_BUFFER, false, true, true, true );
    }

    void SurfaceMeshData::UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
    {
        SE::Core::Primitives::VertexBufferData l_Plane = SE::Core::Primitives::CreatePlane( math::ivec2{ 32, 32 } );
        std::vector<SimpleVertexData>          l_PlaneVertexData =
            ToSimpleVertex( l_Plane.Vertices, math::mat4( 1.0f ) ); //(l_Plane.Vertices.size());

        Mesh.Vertices = New<VkGpuBuffer>( a_GraphicContext, l_PlaneVertexData, eBufferType::VERTEX_BUFFER, false, true, true, true );
        Mesh.Indices  = New<VkGpuBuffer>( a_GraphicContext, l_Plane.Indices, eBufferType::INDEX_BUFFER, false, true, true, true );
    }

    void SpotlightHelperComponent::UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
    {
        Origin.UpdateMesh( a_GraphicContext );

        SE::Core::Primitives::WireframeVertexBufferData l_Spot =
            SE::Core::Primitives::CreateWireframeCone( Spot.Divisions, Spot.Segments );
        math::mat4 l_Transform = math::Translation( math::vec3{ 0.0f, 0.0f, -1.0f } ) * math::Rotation( 90.0_degf, math::x_axis() );
        ApplyTransform( l_Spot.Vertices, l_Transform );

        Spot.Mesh.Vertices =
            New<VkGpuBuffer>( a_GraphicContext, l_Spot.Vertices, eBufferType::VERTEX_BUFFER, false, true, true, true );
        Spot.Mesh.Indices = New<VkGpuBuffer>( a_GraphicContext, l_Spot.Indices, eBufferType::INDEX_BUFFER, false, true, true, true );
    }

    void FieldOfViewHelperComponent::UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
    {
        SE::Core::Primitives::WireframeVertexBufferData l_Pyramid = SE::Core::Primitives::CreateWireframePyramid( 64, 64 );

        math::mat4 l_Transform = math::Translation( math::vec3{ 0.0f, 0.0f, -1.0f } ) * math::Rotation( 90.0_degf, math::x_axis() );
        ApplyTransform( l_Pyramid.Vertices, l_Transform );

        Outline.Mesh.Vertices =
            New<VkGpuBuffer>( a_GraphicContext, l_Pyramid.Vertices, eBufferType::VERTEX_BUFFER, false, true, true, true );
        Outline.Mesh.Indices =
            New<VkGpuBuffer>( a_GraphicContext, l_Pyramid.Indices, eBufferType::INDEX_BUFFER, false, true, true, true );

        SE::Core::Primitives::VertexBufferData l_Plane = SE::Core::Primitives::CreatePlane( math::ivec2{ 32, 32 } );
        std::vector<SimpleVertexData>          l_PlaneVertexData( l_Plane.Vertices.size() );
        for( uint32_t i = 0; i < l_Plane.Vertices.size(); i++ )
        {
            l_PlaneVertexData[i].Position = math::vec4( math::vec2( l_Plane.Vertices[i].Position ), -1.0f, 1.0f );
            l_PlaneVertexData[i].Normal   = l_Plane.Vertices[i].Normal;
        }

        OuterLimit.Mesh.Vertices =
            New<VkGpuBuffer>( a_GraphicContext, l_PlaneVertexData, eBufferType::VERTEX_BUFFER, false, true, true, true );
        OuterLimit.Mesh.Indices =
            New<VkGpuBuffer>( a_GraphicContext, l_Plane.Indices, eBufferType::INDEX_BUFFER, false, true, true, true );
    }

} // namespace SE::Core::EntityComponentSystem::Components