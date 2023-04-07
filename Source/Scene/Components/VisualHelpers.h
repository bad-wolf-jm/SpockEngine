#pragma once

#include "Core/EntityCollection/Collection.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"

#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Scene/Components.h"

namespace SE::Core::EntityComponentSystem::Components
{

    struct VisualHelperComponent
    {
    };

    template <typename _VertexType, typename _IndexType>
    struct MeshData
    {
        using VertexType = _VertexType;
        using IndexType  = _IndexType;

        Ref<VkGpuBuffer> Vertices;
        Ref<VkGpuBuffer> Indices;

        MeshData()                   = default;
        MeshData( const MeshData & ) = default;
    };

    struct PyramidMeshData
    {
        MeshData<math::vec3, uint32_t> Mesh;

        PyramidMeshData()                          = default;
        PyramidMeshData( const PyramidMeshData & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext );
    };

    struct SurfaceMeshData
    {
        MeshData<SimpleVertexData, uint32_t> Mesh;

        SurfaceMeshData()                          = default;
        SurfaceMeshData( const SurfaceMeshData & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext );
    };

    struct CubeMeshData
    {
        float                                SideLength = 1.0f;
        MeshData<SimpleVertexData, uint32_t> Mesh;

        CubeMeshData()                       = default;
        CubeMeshData( const CubeMeshData & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext );
    };

    struct ConeMeshData
    {
        uint32_t Segments  = 32;
        uint32_t Divisions = 5;

        MeshData<math::vec3, uint32_t> Mesh;

        ConeMeshData()                       = default;
        ConeMeshData( const ConeMeshData & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext );
    };

    struct ArrowMeshData
    {
        float                                Length   = 1.0f;
        uint32_t                             Segments = 32;
        MeshData<SimpleVertexData, uint32_t> Mesh;

        ArrowMeshData()                        = default;
        ArrowMeshData( const ArrowMeshData & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext );
    };

    struct CircleMeshData
    {
        float    Radius   = 1.0f;
        uint32_t Segments = 64;

        MeshData<math::vec3, uint32_t> Mesh;

        CircleMeshData()                         = default;
        CircleMeshData( const CircleMeshData & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext );
    };

    struct AxesComponent
    {
        math::vec3 XAxisColor = math::vec3{ 0.8f, 0.1f, 0.15f };
        math::vec3 YAxisColor = math::vec3{ 0.3f, 0.8f, 0.3f };
        math::vec3 ZAxisColor = math::vec3{ 0.1f, 0.25f, 0.8f };

        math::vec3 OriginColor = math::vec3{ 0.6f, 0.6f, 0.6f };

        ArrowMeshData AxisArrow;
        CubeMeshData  Origin;

        AxesComponent()                        = default;
        AxesComponent( const AxesComponent & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
        {
            AxisArrow.UpdateMesh( a_GraphicContext );
            Origin.UpdateMesh( a_GraphicContext );
        }
    };

    struct PointLightHelperComponent
    {
        sLightComponent LightData;

        CubeMeshData   Origin;
        CircleMeshData AxisCircle;

        PointLightHelperComponent()                                    = default;
        PointLightHelperComponent( const PointLightHelperComponent & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
        {
            Origin.UpdateMesh( a_GraphicContext );
            AxisCircle.UpdateMesh( a_GraphicContext );
        }
    };

    struct DirectionalLightHelperComponent
    {
        sLightComponent LightData;

        ArrowMeshData Direction;
        CubeMeshData  Origin;

        DirectionalLightHelperComponent()                                          = default;
        DirectionalLightHelperComponent( const DirectionalLightHelperComponent & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
        {
            Origin.UpdateMesh( a_GraphicContext );
            Direction.UpdateMesh( a_GraphicContext );
        }
    };

    struct SpotlightHelperComponent
    {
        sLightComponent LightData;

        CubeMeshData Origin;
        ConeMeshData Spot;

        SpotlightHelperComponent()                                   = default;
        SpotlightHelperComponent( const SpotlightHelperComponent & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext );
    };

    struct FieldOfViewHelperComponent
    {
        math::vec3 LookAtDirection;
        float      Rotation;
        float      Width;
        float      Height;

        PyramidMeshData Outline;
        SurfaceMeshData OuterLimit;

        FieldOfViewHelperComponent()                                     = default;
        FieldOfViewHelperComponent( const FieldOfViewHelperComponent & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext );
    };

    struct CameraHelperComponent
    {
        sCameraComponent CameraData;

        CubeMeshData               Origin;
        FieldOfViewHelperComponent FieldOfView;

        CameraHelperComponent()                                = default;
        CameraHelperComponent( const CameraHelperComponent & ) = default;

        void UpdateMesh( Ref<VkGraphicContext> a_GraphicContext )
        {
            Origin.UpdateMesh( a_GraphicContext );
            FieldOfView.UpdateMesh( a_GraphicContext );
        }
    };

} // namespace SE::Core::EntityComponentSystem::Components