#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"

#include "Graphics/Vulkan/ARenderContext.h"
#include "Graphics/Vulkan/DescriptorSet.h"

#include "Graphics/Vulkan/GraphicsPipeline.h"
#include "Graphics/Vulkan/VkAbstractRenderPass.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

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
        VisualHelperMeshRenderer( Ref<VkGraphicContext> a_GraphicContext, VisualHelperMeshRendererCreateInfo a_CreateInfo );

        ~VisualHelperMeshRenderer() = default;

        void Render( math::mat4 a_Model, math::mat4 a_View, math::mat4 a_Projection, math::vec3 a_Color,
                     Ref<VkGpuBuffer> a_VertexBuffer, Ref<VkGpuBuffer> a_IndexBuffer, ARenderContext &aRenderContext );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();
    };

} // namespace SE::Graphics
