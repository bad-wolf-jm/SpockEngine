#pragma once
#include "Core/Memory.h"

#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//DeferredRenderContext.h"

#include "Core/GraphicContext//Texture2D.h"
#include "Core/GraphicContext//TextureCubemap.h"
#include "Core/Vulkan/VkImage.h"
#include "Core/Vulkan/VkRenderPass.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "CoordinateGridRenderer.h"
#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"
#include "VisualHelperRenderer.h"
#include "SceneRenderData.h"

namespace LTSE::Core
{

    class DeferredSceneRenderer
    {
      public:
        WorldMatrices  View;
        CameraSettings Settings;
        bool           RenderCoordinateGrid = true;
        bool           RenderGizmos         = false;
        bool           GrayscaleRendering   = false;

      public:
        DeferredSceneRenderer() = default;
        DeferredSceneRenderer( Ref<Scene> aWorld, DeferredRenderContext &aRenderContext );

        ~DeferredSceneRenderer() = default;

        void Render( DeferredRenderContext &aRenderContext );

        Ref<Scene> mWorld = nullptr;

      protected:
        MeshRendererCreateInfo GetRenderPipelineCreateInfo(
            DeferredRenderContext &aRenderContext, sMaterialShaderComponent &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo(
            DeferredRenderContext &aRenderContext, sParticleShaderComponent &aPipelineSpecification );

        MeshRenderer &GetRenderPipeline( DeferredRenderContext &aRenderContext, sMaterialShaderComponent &aPipelineSpecification );
        MeshRenderer &GetRenderPipeline( DeferredRenderContext &aRenderContext, MeshRendererCreateInfo const &aPipelineSpecification );

        ParticleSystemRenderer &GetRenderPipeline( DeferredRenderContext &aRenderContext, sParticleShaderComponent &aPipelineSpecification );

      protected:
        GraphicContext mGraphicContext;

        Ref<CoordinateGridRenderer> mCoordinateGridRenderer = nullptr;
        Ref<VisualHelperRenderer>   mVisualHelperRenderer   = nullptr;

        Ref<Buffer> mCameraUniformBuffer    = nullptr;
        Ref<Buffer> mShaderParametersBuffer = nullptr;

        Ref<DescriptorSetLayout> mCameraSetLayout  = nullptr;
        Ref<DescriptorSetLayout> mNodeSetLayout    = nullptr;
        Ref<DescriptorSetLayout> mTextureSetLayout = nullptr;

        Ref<DescriptorSet> mSceneDescriptors = nullptr;
        Ref<DescriptorSet> mNodeDescriptors  = nullptr;

        Ref<Graphics::Texture2D> mEmptyTexture = nullptr;

        std::unordered_map<MeshRendererCreateInfo, MeshRenderer, MeshRendererCreateInfoHash> mMeshRenderers = {};
        std::unordered_map<ParticleRendererCreateInfo, ParticleSystemRenderer, ParticleSystemRendererCreateInfoHash>
            mParticleRenderers = {};

        std::unordered_map<Entity, Ref<DescriptorSet>> mMaterials = {};

      protected:
        void UpdateDescriptorSets( DeferredRenderContext &aRenderContext );
    };

} // namespace LTSE::Core