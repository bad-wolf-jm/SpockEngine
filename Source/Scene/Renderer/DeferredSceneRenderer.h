#pragma once

#include "Core/GraphicContext/ARenderContext.h"
#include "Core/GraphicContext/ARenderTarget.h"
#include "Core/GraphicContext/DescriptorSet.h"
#include "Core/Memory.h"

#include "ASceneRenderer.h"
#include "CoordinateGridRenderer.h"
#include "DeferredLightingRenderer.h"
#include "MeshRenderer.h"
#include "SceneRenderData.h"
#include "VisualHelperRenderer.h"

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

      public:
        DeferredRenderer() = default;
        DeferredRenderer( GraphicContext aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~DeferredRenderer() = default;

        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

        void Update( Ref<Scene> aWorld );
        void Render();

        Ref<VkTexture2D> GetOutputImage();

        MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification );
        MeshRenderer              &GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification );
        MeshRenderer              &GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification );
        ParticleSystemRenderer    &GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification );

      private:
        void UpdateDescriptorSets();

      private:
        ARenderContext           mGeometryContext{};
        Ref<DescriptorSetLayout> mGeometryCameraLayout = nullptr;
        Ref<DescriptorSet>       mGeometryPassCamera   = nullptr;

        ARenderContext mLightingContext{};

        Ref<ARenderTarget> mGeometryRenderTarget = nullptr;

        Ref<VkGpuBuffer> mCameraUniformBuffer    = nullptr;
        Ref<VkGpuBuffer> mShaderParametersBuffer = nullptr;

        Ref<DescriptorSetLayout> mLightingTextureLayout = nullptr;
        Ref<DescriptorSet>       mLightingPassTextures  = nullptr;
        Ref<DescriptorSetLayout> mLightingCameraLayout  = nullptr;
        Ref<DescriptorSet>       mLightingPassCamera    = nullptr;

        Ref<ARenderTarget> mLightingRenderTarget = nullptr;

        DeferredLightingRenderer mLightingRenderer;

        Ref<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;
        Ref<VisualHelperRenderer>   mVisualHelperRenderer   = nullptr;

        std::unordered_map<MeshRendererCreateInfo, MeshRenderer, MeshRendererCreateInfoHash> mMeshRenderers = {};
        std::unordered_map<ParticleRendererCreateInfo, ParticleSystemRenderer, ParticleSystemRendererCreateInfoHash>
            mParticleRenderers = {};
    };
} // namespace SE::Core