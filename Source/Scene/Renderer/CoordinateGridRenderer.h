#pragma once

#include "Core/Memory.h"

#include "Core/GraphicContext//Buffer.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//RenderContext.h"

#include "Scene/VertexData.h"

#include "Core/Vulkan/VkRenderPass.h"

#include "SceneRenderPipeline.h"

namespace LTSE::Core
{

    using namespace LTSE::Graphics;

    struct CameraViewUniforms
    {
        math::mat4 View;
        math::mat4 Projection;
    };

    struct CoordinateGridRendererCreateInfo
    {
        Ref<LTSE::Graphics::Internal::sVkRenderPassObject> RenderPass = nullptr;
    };

    class CoordinateGridRenderer : public LTSE::Core::SceneRenderPipeline<EmptyVertexData>
    {
      public:
        CoordinateGridRenderer(
            GraphicContext &mGraphicContext, RenderContext &a_RenderContext, CoordinateGridRendererCreateInfo a_CreateInfo );
        ~CoordinateGridRenderer() = default;

        void Render( math::mat4 a_Projection, math::mat4 a_View, RenderContext &aRenderContext );

        CoordinateGridRendererCreateInfo Spec;
        Ref<DescriptorSetLayout>         PipelineLayout;

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

      private:
        Ref<Buffer>        m_CameraBuffer      = nullptr;
        Ref<DescriptorSet> m_CameraDescriptors = nullptr;
    };

} // namespace LTSE::Core