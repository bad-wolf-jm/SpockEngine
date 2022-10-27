#pragma once
#include "Core/Memory.h"

#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//RenderContext.h"

#include "Core/GraphicContext//Texture2D.h"
#include "Core/GraphicContext//TextureCubemap.h"
#include "Graphics/Implementation/Vulkan/VkImage.h"
#include "Graphics/Implementation/Vulkan/VkRenderPass.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "CoordinateGridRenderer.h"
#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"
#include "VisualHelperRenderer.h"

namespace LTSE::Core
{

    using namespace LTSE::Graphics;
    using namespace LTSE::Core::EntityComponentSystem::Components;

#define MAX_NUM_LIGHTS 64

    struct DirectionalLightData
    {
        alignas( 16 ) math::vec3 Direction = math::vec3( 0.0f );
        alignas( 16 ) math::vec3 Color     = math::vec3( 0.0f );
        alignas( 4 ) float Intensity       = 0.0f;

        DirectionalLightData()  = default;
        ~DirectionalLightData() = default;

        DirectionalLightData( const DirectionalLightData & ) = default;
        DirectionalLightData( const sDirectionalLightComponent &a_Spec, math::mat4 a_Transform );
    };

    struct PointLightData
    {
        alignas( 16 ) math::vec3 WorldPosition = math::vec3( 0.0f );
        alignas( 16 ) math::vec3 Color         = math::vec3( 0.0f );
        alignas( 4 ) float Intensity           = 0.0f;

        PointLightData()  = default;
        ~PointLightData() = default;

        PointLightData( const PointLightData & ) = default;
        PointLightData( const sPointLightComponent &a_Spec, math::mat4 a_Transform );
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
        SpotlightData( const sSpotlightComponent &a_Spec, math::mat4 a_Transform );
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

    struct CameraSettings
    {
        float Exposure                             = 4.5f;
        float Gamma                                = 2.2f;
        float AmbientLightIntensity                = 0.0001;
        alignas( 16 ) math::vec4 AmbientLightColor = math::vec4{ 1.0f, 1.0f, 1.0f, 0.0f };
        float DebugViewInputs                      = 0.0f;
        float DebugViewEquation                    = 0.0f;
    };

#define MAX_NUM_JOINTS 512
    struct NodeMatrixDataComponent
    {
        math::mat4 Transform = math::mat4( 1.0f );
        math::mat4 Joints[MAX_NUM_JOINTS]{};
        float      JointCount = 0;
    };

    struct NodeDescriptorComponent
    {
        Ref<Buffer>        UniformBuffer = nullptr;
        Ref<DescriptorSet> Descriptors   = nullptr;

        NodeDescriptorComponent()                                  = default;
        NodeDescriptorComponent( const NodeDescriptorComponent & ) = default;
    };

    struct MaterialDescriptorComponent
    {
        Ref<DescriptorSet> Descriptors = nullptr;

        MaterialDescriptorComponent()                                      = default;
        MaterialDescriptorComponent( const MaterialDescriptorComponent & ) = default;
    };

    class SceneRenderer
    {
      public:
        WorldMatrices  View;
        CameraSettings Settings;
        bool           RenderCoordinateGrid = true;
        bool           RenderGizmos         = false;
        bool           GrayscaleRendering   = false;

      public:
        SceneRenderer() = default;
        SceneRenderer(
            Ref<Scene> a_World, RenderContext &a_RenderContext, Ref<LTSE::Graphics::Internal::sVkRenderPassObject> a_RenderPass );

        ~SceneRenderer() = default;

        void Render( RenderContext &aRenderContext );

        Ref<Scene> m_World = nullptr;

      protected:
        MeshRendererCreateInfo GetRenderPipelineCreateInfo(
            RenderContext &aRenderContext, sMaterialShaderComponent &a_PipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo(
            RenderContext &aRenderContext, sParticleShaderComponent &a_PipelineSpecification );

        MeshRenderer &GetRenderPipeline( RenderContext &aRenderContext, sMaterialShaderComponent &a_PipelineSpecification );
        MeshRenderer &GetRenderPipeline( RenderContext &aRenderContext, MeshRendererCreateInfo const &a_PipelineSpecification );

        ParticleSystemRenderer &GetRenderPipeline( RenderContext &aRenderContext, sParticleShaderComponent &a_PipelineSpecification );

      protected:
        GraphicContext mGraphicContext;

        Ref<CoordinateGridRenderer> m_CoordinateGridRenderer = nullptr;
        Ref<VisualHelperRenderer>   m_VisualHelperRenderer   = nullptr;

        Ref<Buffer> m_CameraUniformBuffer    = nullptr;
        Ref<Buffer> m_ShaderParametersBuffer = nullptr;

        Ref<LTSE::Graphics::Internal::sVkRenderPassObject> m_RenderPass = nullptr;

        Ref<DescriptorSetLayout> m_CameraSetLayout  = nullptr;
        Ref<DescriptorSetLayout> m_NodeSetLayout    = nullptr;
        Ref<DescriptorSetLayout> m_TextureSetLayout = nullptr;

        Ref<DescriptorSet> m_SceneDescriptors = nullptr;
        Ref<DescriptorSet> m_NodeDescriptors  = nullptr;

        Ref<Graphics::Texture2D> m_EmptyTexture = nullptr;

        std::unordered_map<MeshRendererCreateInfo, MeshRenderer, MeshRendererCreateInfoHash> m_MeshRenderers = {};
        std::unordered_map<ParticleRendererCreateInfo, ParticleSystemRenderer, ParticleSystemRendererCreateInfoHash>
            m_ParticleRenderers = {};

        std::unordered_map<Entity, Ref<DescriptorSet>> m_Materials = {};

      protected:
        void UpdateDescriptorSets( RenderContext &aRenderContext );
    };

} // namespace LTSE::Core