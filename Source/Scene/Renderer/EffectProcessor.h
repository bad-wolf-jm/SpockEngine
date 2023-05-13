#pragma once

#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"

#include "Graphics/Vulkan/VkRenderContext.h"
#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/VkGraphicsPipeline.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

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

    class EffectProcessor : public SE::Core::SceneRenderPipeline<EmptyVertexData>
    {
      public:
        EffectProcessor( Ref<VkGraphicContext> mGraphicContext, VkRenderContext &aRenderContext,
                         EffectProcessorCreateInfo aCreateInfo );
        ~EffectProcessor() = default;

        void Render( Ref<Graphics::VkSampler2D> aImageSampler, VkRenderContext &aRenderContext );

        EffectProcessorCreateInfo Spec;
        Ref<DescriptorSetLayout>  PipelineLayout = nullptr;
        Ref<DescriptorSet>        mTextures       = nullptr;

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

      private:
        Ref<VkGpuBuffer>   mCameraBuffer      = nullptr;
        Ref<DescriptorSet> mCameraDescriptors = nullptr;
    };

} // namespace SE::Core