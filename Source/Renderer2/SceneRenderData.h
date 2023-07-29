#pragma once

#pragma once
#include "Core/Memory.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Scene/Components.h"
// #include "Scene/Scene.h"

// #include "CoordinateGridRenderer.h"
// #include "MeshRenderer.h"
// #include "ParticleSystemRenderer.h"
// #include "VisualHelperRenderer.h"

namespace SE::Core
{

    using namespace SE::Graphics;
    // using namespace SE::Core::EntityComponentSystem::Components;

    enum class eNewLightType : uint8_t
    {
        DIRECTIONAL = 0,
        SPOTLIGHT   = 1,
        POINT_LIGHT = 2
    };

    struct sNewLightComponent
    {
        eNewLightType mType = eNewLightType::POINT_LIGHT;

        bool       mIsOn      = true;
        float      mIntensity = 100.0f;
        math::vec3 mColor     = { 1.0f, 1.0f, 1.0f };
        float      mCone      = 60.0f;

        sNewLightComponent()                             = default;
        sNewLightComponent( const sNewLightComponent & ) = default;
    };

#define MAX_NUM_LIGHTS 64

    struct sDirectionalLightData
    {
        alignas( 16 ) math::vec3 Direction = math::vec3( 0.0f );
        alignas( 16 ) math::vec3 Color     = math::vec3( 0.0f );
        alignas( 4 ) float Intensity       = 0.0f;
        alignas( 16 ) math::mat4 Transform = math::mat4( 0.0f );

        uint32_t mIsOn = 1u;

        sDirectionalLightData()  = default;
        ~sDirectionalLightData() = default;

        sDirectionalLightData( const sDirectionalLightData & ) = default;
        sDirectionalLightData( const sNewLightComponent &a_Spec, math::mat4 a_Transform );
    };

    struct sPointLightData
    {
        alignas( 16 ) math::vec3 WorldPosition = math::vec3( 0.0f );
        alignas( 16 ) math::vec3 Color         = math::vec3( 0.0f );
        alignas( 4 ) float Intensity           = 0.0f;

        uint32_t mIsOn = 1u;

        sPointLightData()  = default;
        ~sPointLightData() = default;

        sPointLightData( const sPointLightData & ) = default;
        sPointLightData( const sNewLightComponent &a_Spec, math::mat4 a_Transform );
    };

    // struct WorldMatrices
    // {
    //     alignas( 16 ) math::mat4 Projection;
    //     alignas( 16 ) math::mat4 ModelFraming;
    //     alignas( 16 ) math::mat4 View;
    //     alignas( 16 ) math::vec3 CameraPosition;

    //     alignas( 4 ) int DirectionalLightCount = 0;
    //     alignas( 16 ) DirectionalLightData DirectionalLights[MAX_NUM_LIGHTS];

    //     alignas( 4 ) int SpotlightCount = 0;
    //     alignas( 16 ) SpotlightData Spotlights[MAX_NUM_LIGHTS];

    //     alignas( 4 ) int PointLightCount = 0;
    //     alignas( 16 ) PointLightData PointLights[MAX_NUM_LIGHTS];

    //     WorldMatrices()  = default;
    //     ~WorldMatrices() = default;

    //     WorldMatrices( const WorldMatrices & ) = default;
    // };

    struct NewShadowMatrices
    {
        math::mat4 mMVP;

        NewShadowMatrices()  = default;
        ~NewShadowMatrices() = default;

        NewShadowMatrices( const NewShadowMatrices & ) = default;
    };

    struct NewOmniShadowMatrices
    {
        math::mat4 mMVP;
        math::vec4 mLightPos;

        NewOmniShadowMatrices()  = default;
        ~NewOmniShadowMatrices() = default;

        NewOmniShadowMatrices( const NewOmniShadowMatrices & ) = default;
    };

    // struct CameraSettings
    // {
    //     float Exposure                             = 4.5f;
    //     float Gamma                                = 2.2f;
    //     float AmbientLightIntensity                = 0.0001;
    //     alignas( 16 ) math::vec4 AmbientLightColor = math::vec4{ 1.0f, 1.0f, 1.0f, 0.0f };
    //     float DebugViewInputs                      = 0.0f;
    //     float DebugViewEquation                    = 0.0f;
    //     float RenderGrayscale                      = 0.0f;
    // };

    // struct sLightGizmo
    // {
    //     eLightType mType;
    //     uint64_t   mLightDataIndex = 0;
    //     math::mat4 mMatrix{};

    //     sLightGizmo()  = default;
    //     ~sLightGizmo() = default;

    //     sLightGizmo( eLightType aType, uint64_t aLightDataIndex, math::mat4 aMatrix );
    // };

    // struct sParticleRenderData
    // {
    //     math::mat4          mModel         = math::mat4( 1.0f );
    //     uint32_t            mParticleCount = 0;
    //     float               mLineWidth     = 1.0f;
    //     float               mParticleSize  = 1.0f;
    //     Ref<IGraphicBuffer> mParticles     = nullptr;

    //     sParticleRenderData( sParticleSystemComponent const &aParticles, sParticleShaderComponent const &aShader );
    // };

} // namespace SE::Core