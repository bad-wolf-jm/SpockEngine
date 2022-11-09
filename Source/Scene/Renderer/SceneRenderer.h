#pragma once
#include "Core/Memory.h"

#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//RenderContext.h"

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

    {
      public:
        WorldMatrices  View;
        CameraSettings Settings;
        bool           RenderCoordinateGrid = true;
        bool           RenderGizmos         = false;
        bool           GrayscaleRendering   = false;

      public:
        SceneRenderer() = default;
        SceneRenderer( Ref<Scene> aWorld, RenderContext &aRenderContext );

        ~SceneRenderer() = default;

        void Render( RenderContext &aRenderContext );

        Ref<Scene> mWorld = nullptr;

      protected:
        MeshRendererCreateInfo GetRenderPipelineCreateInfo(
            RenderContext &aRenderContext, sMaterialShaderComponent &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo(
            RenderContext &aRenderContext, sParticleShaderComponent &aPipelineSpecification );

        MeshRenderer &GetRenderPipeline( RenderContext &aRenderContext, sMaterialShaderComponent &aPipelineSpecification );
        MeshRenderer &GetRenderPipeline( RenderContext &aRenderContext, MeshRendererCreateInfo const &aPipelineSpecification );

        ParticleSystemRenderer &GetRenderPipeline( RenderContext &aRenderContext, sParticleShaderComponent &aPipelineSpecification );

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
        void UpdateDescriptorSets( RenderContext &aRenderContext );
    };

} // namespace LTSE::Core