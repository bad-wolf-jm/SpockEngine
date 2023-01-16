#pragma once
#include "Core/Memory.h"

#include "Graphics/Vulkan/ARenderContext.h"
#include "Graphics/Vulkan/VkRenderTarget.h"
#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "Renderer/ASceneRenderer.h"
#include "Renderer/SceneRenderData.h"

#include "CoordinateGridRenderer.h"
#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"
#include "VisualHelperRenderer.h"

namespace SE::Core
{

    class ForwardSceneRenderer : public ASceneRenderer
    {
      public:
        WorldMatrices  View;
        CameraSettings Settings;
        bool           RenderCoordinateGrid = true;
        bool           RenderGizmos         = false;
        bool           GrayscaleRendering   = false;

      public:
        ForwardSceneRenderer() = default;
        ForwardSceneRenderer( Ref<VkGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~ForwardSceneRenderer() = default;

        Ref<VkTexture2D> GetOutputImage();

        void Update( Ref<Scene> aWorld );
        void Render();

        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

      protected:
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

      protected:
        Ref<VkRenderTarget> mGeometryRenderTarget = nullptr;
        ARenderContext     mGeometryContext{};

        Ref<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;
        Ref<VisualHelperRenderer>   mVisualHelperRenderer   = nullptr;

        Ref<VkGpuBuffer> mCameraUniformBuffer    = nullptr;
        Ref<VkGpuBuffer> mShaderParametersBuffer = nullptr;

        Ref<DescriptorSetLayout> mCameraSetLayout  = nullptr;
        Ref<DescriptorSetLayout> mNodeSetLayout    = nullptr;
        Ref<DescriptorSetLayout> mTextureSetLayout = nullptr;

        Ref<DescriptorSet> mSceneDescriptors = nullptr;
        Ref<DescriptorSet> mNodeDescriptors  = nullptr;

        std::unordered_map<MeshRendererCreateInfo, MeshRenderer, MeshRendererCreateInfoHash> mMeshRenderers = {};
        std::unordered_map<ParticleRendererCreateInfo, ParticleSystemRenderer, ParticleSystemRendererCreateInfoHash>
            mParticleRenderers = {};

        std::unordered_map<Entity, Ref<DescriptorSet>> mMaterials = {};
    };

} // namespace SE::Core