#pragma once

#include "Core/Memory.h"
// #include "Graphics/Vulkan/Ref<IRenderContext>.h"
// #include "Graphics/Vulkan/IDescriptorSet.h"
// #include "Graphics/Vulkan/IRenderTarget.h"

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
    using namespace SE::Graphics;

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

        Ref<ITexture> GetOutputImage();

        MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification );
        MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMeshRenderData &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleRenderData &aPipelineSpecification );

        Ref<MeshRenderer>           GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification );
        Ref<MeshRenderer>           GetRenderPipeline( sMeshRenderData &aPipelineSpecification );
        Ref<MeshRenderer>           GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification );
        Ref<ParticleSystemRenderer> GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification );
        Ref<ParticleSystemRenderer> GetRenderPipeline( sParticleRenderData &aPipelineSpecification );
        Ref<ParticleSystemRenderer> GetRenderPipeline( ParticleRendererCreateInfo &aPipelineSpecification );

      private:
        Ref<IRenderContext> mGeometryContext{};
        // Ref<DescriptorSetLayout> mGeometryCameraLayout = nullptr;
        Ref<IDescriptorSet> mGeometryPassCamera = nullptr;

        Ref<IRenderTarget> mGeometryRenderTarget = nullptr;

        Ref<IGraphicBuffer> mCameraUniformBuffer    = nullptr;
        Ref<IGraphicBuffer> mShaderParametersBuffer = nullptr;

        std::map<std::string, Ref<ISampler2D>> mGeometrySamplers = {};

        // Ref<DescriptorSetLayout> mLightingTextureLayout = nullptr;
        // Ref<DescriptorSetLayout> mLightingCameraLayout  = nullptr;
        // Ref<DescriptorSetLayout> mLightingDirectionalShadowLayout   = nullptr;
        // Ref<DescriptorSetLayout> mLightingSpotlightShadowLayout   = nullptr;
        // Ref<DescriptorSetLayout> mLightingPointLightShadowLayout   = nullptr;
        Ref<IDescriptorSet> mLightingPassTextures              = nullptr;
        Ref<IDescriptorSet> mLightingPassCamera                = nullptr;
        Ref<IDescriptorSet> mLightingPassDirectionalShadowMaps = nullptr;
        Ref<IDescriptorSet> mLightingPassSpotlightShadowMaps   = nullptr;
        Ref<IDescriptorSet> mLightingPassPointLightShadowMaps  = nullptr;

        Ref<DeferredLightingRenderer> mLightingRenderer     = nullptr;
        Ref<IRenderTarget>            mLightingRenderTarget = nullptr;
        Ref<IRenderContext>           mLightingContext{};

        Ref<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;
        Ref<ShadowSceneRenderer>    mShadowSceneRenderer    = nullptr;

        Ref<EffectProcessor> mCopyRenderer     = nullptr;
        Ref<EffectProcessor> mFxaaRenderer     = nullptr;
        Ref<ISampler2D>      mFxaaSampler      = nullptr;
        Ref<IRenderTarget>   mFxaaRenderTarget = nullptr;
        Ref<IRenderContext>  mFxaaContext{};

        std::unordered_map<MeshRendererCreateInfo, Ref<MeshRenderer>, MeshRendererCreateInfoHash> mMeshRenderers = {};
        std::unordered_map<ParticleRendererCreateInfo, Ref<ParticleSystemRenderer>, ParticleSystemRendererCreateInfoHash>
            mParticleRenderers = {};
    };
} // namespace SE::Core