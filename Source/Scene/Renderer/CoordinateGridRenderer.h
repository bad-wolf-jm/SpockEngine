#pragma once

#include "Core/Memory.h"

// #include "Graphics/Vulkan/IGraphicsBuffer.h"
// #include "Graphics/Vulkan/VkRenderContext.h"
// #include "Graphics/Vulkan/DescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Graphics/Vulkan/IGraphicContext.h"
// #include "Graphics/Vulkan/IRenderPass.h"
#include "Graphics/API.h"

#include "Scene/VertexData.h"

// #include "SceneRenderPipeline.h"

namespace SE::Core
{

    using namespace SE::Graphics;

    struct CameraViewUniforms
    {
        math::mat4 View;
        math::mat4 Projection;
    };

    // struct CoordinateGridRendererCreateInfo
    // {
    //     Ref<IRenderPass> RenderPass = nullptr;
    // };

    class CoordinateGridRenderer
    // : public SE::Core::SceneRenderPipeline<EmptyVertexData>
    {
      public:
        CoordinateGridRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext );
        ~CoordinateGridRenderer() = default;

        void Render( math::mat4 aProjection, math::mat4 aView, Ref<IRenderContext> &aRenderContext );

        // CoordinateGridRendererCreateInfo Spec;
        // Ref<DescriptorSetLayout>         PipelineLayout;
        // std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        // std::vector<sPushConstantRange>       GetPushConstantLayout();

      private:
        Ref<IGraphicContext>   mGraphicContext    = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer      = nullptr;
        Ref<IDescriptorSet>    mCameraDescriptors = nullptr;
        Ref<IGraphicsPipeline> mPipeline          = nullptr;
    };

} // namespace SE::Core