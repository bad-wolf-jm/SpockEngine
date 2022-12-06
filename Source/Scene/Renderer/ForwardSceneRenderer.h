#pragma once
#include "Core/Memory.h"

#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext/ARenderContext.h"
#include "Core/GraphicContext/ARenderTarget.h"

#include "Core/GraphicContext//Texture2D.h"
#include "Core/GraphicContext//TextureCubemap.h"
#include "Core/Vulkan/VkImage.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "ASceneRenderer.h"
#include "CoordinateGridRenderer.h"
#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"
#include "SceneRenderData.h"
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
        ForwardSceneRenderer( GraphicContext aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~ForwardSceneRenderer() = default;

        Ref<sVkFramebufferImage> GetOutputImage();

        void Update( Ref<Scene> aWorld );
        void Render();

        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

      protected:
        MeshRendererCreateInfo     GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification );
        ParticleRendererCreateInfo GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification );

        MeshRenderer           &GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification );
        MeshRenderer           &GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification );
        ParticleSystemRenderer &GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification );

      protected:
        Ref<ARenderTarget> mGeometryRenderTarget = nullptr;
        ARenderContext     mGeometryContext{};

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
        void UpdateDescriptorSets();
    };

} // namespace SE::Core