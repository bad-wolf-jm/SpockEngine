#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

// #include "Graphics/Vulkan/VkGpuBuffer.h"
// #include "Graphics/Vulkan/VkRenderContext.h"
// #include "Graphics/Vulkan/DescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Graphics/Vulkan/IGraphicContext.h"
// #include "Graphics/Vulkan/VkRenderPass.h"
#include "Graphics/API.h"

#include "Scene/VertexData.h"

// #include "SceneRenderPipeline.h"

namespace SE::Graphics
{
    using namespace math;
    using namespace SE::Core;
    namespace fs = std::filesystem;

    struct VisualHelperLineRendererCreateInfo
    {
        float                            LineWidth  = 1.0f;
        Ref<sVkAbstractRenderPassObject> RenderPass = nullptr;
    };

    class VisualHelperLineRenderer// : public SceneRenderPipeline<PositionData>
    {
      public:
        struct CameraViewUniforms
        {
            mat4 ModelMatrix;
            mat4 ViewMatrix;
            mat4 ProjectionMatrix;
            vec4 Color;

            CameraViewUniforms()                             = default;
            CameraViewUniforms( const CameraViewUniforms & ) = default;
        };

        VisualHelperLineRendererCreateInfo Spec;

        VisualHelperLineRenderer() = default;
        VisualHelperLineRenderer( Ref<IGraphicContext> aGraphicContext, VisualHelperLineRendererCreateInfo aCreateInfo );
        ~VisualHelperLineRenderer() = default;

        void Render( math::mat4 aModel, math::mat4 aView, math::mat4 aProjection, math::vec3 aColor, Ref<VkGpuBuffer> aVertexBuffer,
                     Ref<VkGpuBuffer> aIndexBuffer, Ref<iRenderContext> aRenderContext );

        std::vector<Ref<IDescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>        GetPushConstantLayout();
    };

} // namespace SE::Graphics
