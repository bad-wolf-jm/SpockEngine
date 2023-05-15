#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

// #include "Graphics/Vulkan/IGraphicBuffer.h"
// #include "Graphics/Vulkan/VkRenderContext.h"
// #include "Graphics/Vulkan/DescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Graphics/Vulkan/VkRenderPass.h"
// #include "Graphics/Vulkan/IGraphicContext.h"
#include "Graphics/API.h"

#include "Scene/ParticleData.h"
#include "Scene/VertexData.h"

#include "SceneRenderPipeline.h"

namespace SE::Graphics
{

    using namespace math;
    namespace fs = std::filesystem;

    struct VisualHelperMeshRendererCreateInfo
    {
        float                                          LineWidth  = 1.0f;
        Ref<SE::Graphics::sVkAbstractRenderPassObject> RenderPass = nullptr;
    };

    class VisualHelperMeshRenderer : public SE::Core::SceneRenderPipeline<SimpleVertexData>
    {
      public:
        struct CameraViewUniforms
        {
            mat4 Model;
            mat4 View;
            mat4 Projection;
            vec4 Color;
        };

        VisualHelperMeshRendererCreateInfo Spec;

        VisualHelperMeshRenderer() = default;
        VisualHelperMeshRenderer( Ref<IGraphicContext> aGraphicContext, VisualHelperMeshRendererCreateInfo aCreateInfo );

        ~VisualHelperMeshRenderer() = default;

        void Render( math::mat4 aModel, math::mat4 aView, math::mat4 aProjection, math::vec3 aColor, Ref<IGraphicBuffer> aVertexBuffer,
                     Ref<IGraphicBuffer> aIndexBuffer, Ref<iRenderContext> aRenderContext );

        std::vector<Ref<IDescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>        GetPushConstantLayout();
    };

} // namespace SE::Graphics
