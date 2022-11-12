#pragma once
#include "Core/Memory.h"

#include "Core/GraphicContext//DeferredLightingRenderContext.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicsPipeline.h"


#include "Core/GraphicContext//Texture2D.h"
#include "Core/GraphicContext//TextureCubemap.h"
#include "Core/Vulkan/VkImage.h"
#include "Core/Vulkan/VkRenderPass.h"

#include "Scene/Components.h"
#include "Scene/Scene.h"

#include "CoordinateGridRenderer.h"
#include "DeferredLightingRenderer.h"
#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"
#include "SceneRenderData.h"
#include "VisualHelperRenderer.h"


namespace LTSE::Core
{

    class DeferredLightingPass
    {
      public:
        WorldMatrices  View;
        CameraSettings Settings;
        bool           RenderCoordinateGrid = true;
        bool           RenderGizmos         = false;
        bool           GrayscaleRendering   = false;

      public:
        DeferredLightingPass() = default;
        DeferredLightingPass( Ref<Scene> aWorld, DeferredLightingRenderContext &aRenderContext );

        ~DeferredLightingPass() = default;

        void Render(  Ref<DescriptorSet> aTextures, DeferredLightingRenderContext &aRenderContext );

        Ref<DescriptorSetLayout> GetTextureSetLayout() { return mRenderer.GetTextureSetLayout( mGraphicContext ); }

        Ref<Scene> mWorld = nullptr;

      protected:
        DeferredLightingRenderer mRenderer;

      protected:
        GraphicContext mGraphicContext;

        Ref<Buffer> mCameraUniformBuffer    = nullptr;
        Ref<Buffer> mShaderParametersBuffer = nullptr;

        Ref<DescriptorSetLayout> mCameraSetLayout  = nullptr;
        Ref<DescriptorSetLayout> mTextureSetLayout = nullptr;

        Ref<DescriptorSet> mSceneDescriptors = nullptr;
        Ref<DescriptorSet> mNodeDescriptors  = nullptr;

      protected:
        void UpdateDescriptorSets( DeferredLightingRenderContext &aRenderContext );
    };

} // namespace LTSE::Core