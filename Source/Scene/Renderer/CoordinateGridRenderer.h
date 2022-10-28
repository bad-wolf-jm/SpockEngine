#pragma once

#include "Core/Memory.h"

#include "Graphics/API/Buffer.h"
#include "Graphics/API/DescriptorSet.h"
#include "Graphics/API/GraphicContext.h"
#include "Graphics/API/GraphicsPipeline.h"
#include "Graphics/API/RenderContext.h"

#include "Scene/VertexData.h"

#include "Graphics/Implementation/Vulkan/VkRenderPass.h"

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
        Ref<Buffer>        mCameraBuffer      = nullptr;
        Ref<DescriptorSet> mCameraDescriptors = nullptr;
    };

} // namespace LTSE::Core