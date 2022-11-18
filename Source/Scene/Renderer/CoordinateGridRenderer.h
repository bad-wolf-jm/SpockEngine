#pragma once

#include "Core/Memory.h"

#include "Core/GraphicContext//Buffer.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//ARenderContext.h"

#include "Scene/VertexData.h"

#include "Core/Vulkan/VkRenderPass.h"

#include "SceneRenderPipeline.h"

namespace SE::Core
{

    using namespace SE::Graphics;

    struct CameraViewUniforms
    {
        math::mat4 View;
        math::mat4 Projection;
    };

    struct CoordinateGridRendererCreateInfo
    {
        Ref<SE::Graphics::Internal::sVkAbstractRenderPassObject> RenderPass = nullptr;
    };

    class CoordinateGridRenderer : public SE::Core::SceneRenderPipeline<EmptyVertexData>
    {
      public:
        CoordinateGridRenderer(
            GraphicContext &mGraphicContext, ARenderContext &a_RenderContext, CoordinateGridRendererCreateInfo a_CreateInfo );
        ~CoordinateGridRenderer() = default;

        void Render( math::mat4 a_Projection, math::mat4 a_View, ARenderContext &aRenderContext );

        CoordinateGridRendererCreateInfo Spec;
        Ref<DescriptorSetLayout>         PipelineLayout;

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

      private:
        Ref<Buffer>        mCameraBuffer      = nullptr;
        Ref<DescriptorSet> mCameraDescriptors = nullptr;
    };

} // namespace SE::Core