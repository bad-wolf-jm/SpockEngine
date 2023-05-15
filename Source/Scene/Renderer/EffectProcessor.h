#pragma once

#include "Core/Memory.h"

// #include "Graphics/Vulkan/IGraphicBuffer.h"
// #include "Graphics/Vulkan/VkRenderContext.h"
// #include "Graphics/Vulkan/IDescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Graphics/Vulkan/IGraphicContext.h"
#include "Graphics/API.h"

#include "Scene/VertexData.h"

#include "Graphics/Vulkan/VkRenderPass.h"

#include "SceneRenderPipeline.h"

namespace SE::Core
{

    using namespace SE::Graphics;

    struct EffectProcessorCreateInfo
    {
        std::string mVertexShader   = "";
        std::string mFragmentShader = "";

        Ref<VkRenderPass> RenderPass = nullptr;
    };

    class EffectProcessor // : public SE::Core::SceneRenderPipeline<EmptyVertexData>
    {
      public:
        EffectProcessor( Ref<IGraphicContext> mGraphicContext, Ref<IRenderContext> aRenderContext,
                         EffectProcessorCreateInfo aCreateInfo );
        ~EffectProcessor() = default;

        void Render( Ref<Graphics::VkSampler2D> aImageSampler, Ref<IRenderContext> aRenderContext );

        EffectProcessorCreateInfo Spec;
        Ref<IDescriptorSetLayout> PipelineLayout = nullptr;
        Ref<IDescriptorSet>       mTextures      = nullptr;

        std::vector<Ref<IDescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>        GetPushConstantLayout();

      private:
        Ref<IGraphicContext>   mGraphicContext    = nullptr;
        Ref<IGraphicsPipeline> mPipeline          = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer      = nullptr;
        Ref<IDescriptorSet>    mCameraDescriptors = nullptr;
    };

} // namespace SE::Core