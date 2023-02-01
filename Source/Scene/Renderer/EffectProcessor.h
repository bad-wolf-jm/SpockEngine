#pragma once

#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"

#include "Graphics/Vulkan/ARenderContext.h"
#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Scene/VertexData.h"

#include "Graphics/Vulkan/VkAbstractRenderPass.h"

#include "SceneRenderPipeline.h"

namespace SE::Core
{

    using namespace SE::Graphics;

    // struct sCameraViewUniforms
    // {
    //     math::mat4 View;
    //     math::mat4 Projection;
    // };

    struct EffectProcessorCreateInfo
    {
        std::string mVertexShader   = "";
        std::string mFragmentShader = "";

        Ref<sVkAbstractRenderPassObject> RenderPass = nullptr;
    };

    class EffectProcessor : public SE::Core::SceneRenderPipeline<EmptyVertexData>
    {
      public:
        EffectProcessor( Ref<VkGraphicContext> mGraphicContext, ARenderContext &aRenderContext,
                         EffectProcessorCreateInfo aCreateInfo );
        ~EffectProcessor() = default;

        void Render( ARenderContext &aRenderContext );

        EffectProcessorCreateInfo Spec;
        Ref<DescriptorSetLayout>  PipelineLayout;

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

      private:
        Ref<VkGpuBuffer>   mCameraBuffer      = nullptr;
        Ref<DescriptorSet> mCameraDescriptors = nullptr;
    };

} // namespace SE::Core