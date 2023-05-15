#pragma once

#include "Core/Memory.h"
// #include "Graphics/Vulkan/VkRenderContext.h"
// #include "Graphics/Vulkan/IDescriptorSet.h"
// #include "Graphics/Vulkan/VkRenderTarget.h"

#include "Graphics/API.h"

#include "Renderer/ASceneRenderer.h"
#include "Renderer/SceneRenderData.h"

#include "CoordinateGridRenderer.h"
#include "DeferredLightingRenderer.h"
#include "EffectProcessor.h"
#include "MeshRenderer.h"
#include "ShadowSceneRenderer.h"

namespace SE::Core
{
    class DeferredRenderer : public ASceneRenderer
    {
      public:
        WorldMatrices  View;
        CameraSettings Settings;

        bool mRenderCoordinateGrid = true;
        bool mRenderGizmos         = false;
        bool mGrayscaleRendering   = false;
        bool mUseFXAA              = false;

      public:
        DeferredRenderer() = default;
        DeferredRenderer( Ref<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~DeferredRenderer() = default;

        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

        void Update( Ref<Scene> aWorld );
        void Render();

        Ref<VkTexture2D> GetOutputImage();

        MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification );
        MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMeshRenderData &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleRenderData &aPipelineSpecification );

        MeshRenderer           &GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification );
        MeshRenderer           &GetRenderPipeline( sMeshRenderData &aPipelineSpecification );
        MeshRenderer           &GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification );
        ParticleSystemRenderer &GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification );
        ParticleSystemRenderer &GetRenderPipeline( sParticleRenderData &aPipelineSpecification );
        ParticleSystemRenderer &GetRenderPipeline( ParticleRendererCreateInfo &aPipelineSpecification );

      private:
        VkRenderContext mGeometryContext{};
        // Ref<DescriptorSetLayout> mGeometryCameraLayout = nullptr;
        Ref<IDescriptorSet> mGeometryPassCamera = nullptr;

        VkRenderContext mLightingContext{};

        Ref<VkRenderTarget> mGeometryRenderTarget = nullptr;

        Ref<VkGpuBuffer> mCameraUniformBuffer    = nullptr;
        Ref<VkGpuBuffer> mShaderParametersBuffer = nullptr;

        std::map<std::string, Ref<Graphics::VkSampler2D>> mGeometrySamplers = {};

        // Ref<DescriptorSetLayout> mLightingTextureLayout = nullptr;
        Ref<IDescriptorSet> mLightingPassTextures = nullptr;
        // Ref<DescriptorSetLayout> mLightingCameraLayout  = nullptr;
        Ref<IDescriptorSet> mLightingPassCamera = nullptr;
        // Ref<DescriptorSetLayout> mLightingDirectionalShadowLayout   = nullptr;
        Ref<IDescriptorSet> mLightingPassDirectionalShadowMaps = nullptr;
        // Ref<DescriptorSetLayout> mLightingSpotlightShadowLayout   = nullptr;
        Ref<IDescriptorSet> mLightingPassSpotlightShadowMaps = nullptr;
        // Ref<DescriptorSetLayout> mLightingPointLightShadowLayout   = nullptr;
        Ref<IDescriptorSet> mLightingPassPointLightShadowMaps = nullptr;

        Ref<VkRenderTarget> mLightingRenderTarget = nullptr;

        DeferredLightingRenderer mLightingRenderer;

        Ref<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;
        Ref<ShadowSceneRenderer>    mShadowSceneRenderer    = nullptr;

        Ref<EffectProcessor>       mCopyRenderer     = nullptr;
        Ref<EffectProcessor>       mFxaaRenderer     = nullptr;
        Ref<Graphics::VkSampler2D> mFxaaSampler      = nullptr;
        Ref<VkRenderTarget>        mFxaaRenderTarget = nullptr;
        VkRenderContext            mFxaaContext{};

        std::unordered_map<MeshRendererCreateInfo, MeshRenderer, MeshRendererCreateInfoHash> mMeshRenderers = {};
        std::unordered_map<ParticleRendererCreateInfo, ParticleSystemRenderer, ParticleSystemRendererCreateInfoHash>
            mParticleRenderers = {};
    };
} // namespace SE::Core