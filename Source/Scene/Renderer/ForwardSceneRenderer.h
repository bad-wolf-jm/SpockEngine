#pragma once
#include "Core/Memory.h"

// #include "Graphics/Vulkan/Ref<IRenderContext>.h"
// #include "Graphics/Vulkan/IRenderTarget.h"
// #include "Graphics/Vulkan/IDescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
#include "Graphics/API.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "Renderer/ASceneRenderer.h"
#include "Renderer/SceneRenderData.h"

#include "CoordinateGridRenderer.h"
#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"

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
        ForwardSceneRenderer( Ref<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~ForwardSceneRenderer() = default;

        Ref<ITexture> GetOutputImage();

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
        Ref<IRenderTarget>  mGeometryRenderTarget = nullptr;
        Ref<IRenderContext> mGeometryContext{};

        Ref<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;

        Ref<IGraphicBuffer> mCameraUniformBuffer    = nullptr;
        Ref<IGraphicBuffer> mShaderParametersBuffer = nullptr;

        Ref<IDescriptorSetLayout> mCameraSetLayout  = nullptr;
        Ref<IDescriptorSetLayout> mNodeSetLayout    = nullptr;
        Ref<IDescriptorSetLayout> mTextureSetLayout = nullptr;

        Ref<IDescriptorSet> mSceneDescriptors = nullptr;
        Ref<IDescriptorSet> mNodeDescriptors  = nullptr;

        std::unordered_map<MeshRendererCreateInfo, MeshRenderer, MeshRendererCreateInfoHash> mMeshRenderers = {};
        std::unordered_map<ParticleRendererCreateInfo, ParticleSystemRenderer, ParticleSystemRendererCreateInfoHash>
            mParticleRenderers = {};

        std::unordered_map<Entity, Ref<IDescriptorSet>> mMaterials = {};
    };

} // namespace SE::Core