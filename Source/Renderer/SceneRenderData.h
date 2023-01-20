#pragma once

#pragma once
#include "Core/Memory.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

// #include "CoordinateGridRenderer.h"
// #include "MeshRenderer.h"
// #include "ParticleSystemRenderer.h"
// #include "VisualHelperRenderer.h"

namespace SE::Core
{

    using namespace SE::Graphics;
    using namespace SE::Core::EntityComponentSystem::Components;

#define MAX_NUM_LIGHTS 64

    struct DirectionalLightData
    {
        alignas( 16 ) math::vec3 Direction = math::vec3( 0.0f );
        alignas( 16 ) math::vec3 Color     = math::vec3( 0.0f );
        alignas( 4 ) float Intensity       = 0.0f;

        DirectionalLightData()  = default;
        ~DirectionalLightData() = default;

        DirectionalLightData( const DirectionalLightData & ) = default;
        DirectionalLightData( const sLightComponent &a_Spec, math::mat4 a_Transform );
    };

    struct PointLightData
    {
        alignas( 16 ) math::vec3 WorldPosition = math::vec3( 0.0f );
        alignas( 16 ) math::vec3 Color         = math::vec3( 0.0f );
        alignas( 4 ) float Intensity           = 0.0f;

        PointLightData()  = default;
        ~PointLightData() = default;

        PointLightData( const PointLightData & ) = default;
        PointLightData( const sLightComponent &a_Spec, math::mat4 a_Transform );
    };

    struct SpotlightData
    {
        alignas( 16 ) math::vec3 WorldPosition   = math::vec3( 0.0f );
        alignas( 16 ) math::vec3 LookAtDirection = math::vec3( 0.0f );
        alignas( 16 ) math::vec3 Color           = math::vec3( 0.0f );
        alignas( 4 ) float Intensity             = 0.0f;
        alignas( 4 ) float Cone                  = 0.0f;

        SpotlightData()  = default;
        ~SpotlightData() = default;

        SpotlightData( const SpotlightData & ) = default;
        SpotlightData( const sLightComponent &a_Spec, math::mat4 a_Transform );
    };

    struct WorldMatrices
    {
        alignas( 16 ) math::mat4 Projection;
        alignas( 16 ) math::mat4 ModelFraming;
        alignas( 16 ) math::mat4 View;
        alignas( 16 ) math::vec3 CameraPosition;

        alignas( 4 ) int DirectionalLightCount = 0;
        alignas( 16 ) DirectionalLightData DirectionalLights[MAX_NUM_LIGHTS];

        alignas( 4 ) int SpotlightCount = 0;
        alignas( 16 ) SpotlightData Spotlights[MAX_NUM_LIGHTS];

        alignas( 4 ) int PointLightCount = 0;
        alignas( 16 ) PointLightData PointLights[MAX_NUM_LIGHTS];

        WorldMatrices()  = default;
        ~WorldMatrices() = default;

        WorldMatrices( const WorldMatrices & ) = default;
    };

    struct ShadowMatrices
    {
        math::mat4 mMVP;

        ShadowMatrices()  = default;
        ~ShadowMatrices() = default;

        ShadowMatrices( const ShadowMatrices & ) = default;
    };

    struct CameraSettings
    {
        float Exposure                             = 4.5f;
        float Gamma                                = 2.2f;
        float AmbientLightIntensity                = 0.0001;
        alignas( 16 ) math::vec4 AmbientLightColor = math::vec4{ 1.0f, 1.0f, 1.0f, 0.0f };
        float DebugViewInputs                      = 0.0f;
        float DebugViewEquation                    = 0.0f;
        float RenderGrayscale                      = 0.0f;
    };

    struct sLightGizmo
    {
        eLightType mType;
        uint64_t   mLightDataIndex = 0;
        math::mat4 mMatrix{};

        sLightGizmo()  = default;
        ~sLightGizmo() = default;

        sLightGizmo( eLightType aType, uint64_t aLightDataIndex, math::mat4 aMatrix );
    };

    struct sMeshRenderData
    {
        // Shader data
        bool     mOpaque     = false;
        bool     mIsTwoSided = false;
        float    mLineWidth  = 1.0f;
        uint32_t mMaterialID = 0;

        // Buffer data
        Ref<VkGpuBuffer> mVertexBuffer = nullptr;
        Ref<VkGpuBuffer> mIndexBuffer  = nullptr;
        uint32_t         mVertexOffset = 0;
        uint32_t         mVertexCount  = 0;
        uint32_t         mIndexOffset  = 0;
        uint32_t         mIndexCount   = 0;

        sMeshRenderData( sStaticMeshComponent const &aMesh, sMaterialComponent const &aMaterialID,
                         sMaterialShaderComponent const &aShader );
    };

    struct sParticleRenderData
    {
        math::mat4       mModel         = math::mat4( 1.0f );
        uint32_t         mParticleCount = 0;
        float            mLineWidth     = 1.0f;
        float            mParticleSize  = 1.0f;
        Ref<VkGpuBuffer> mParticles     = nullptr;

        sParticleRenderData( sParticleSystemComponent const &aParticles, sParticleShaderComponent const &aShader );
    };

} // namespace SE::Core