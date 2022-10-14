#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Types.h"
#include "Core/Memory.h"

#include "Core/GraphicContext//Buffer.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//RenderContext.h"

#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/Vulkan/VkRenderPass.h"

#include "Scene/ParticleData.h"
#include "Scene/VertexData.h"

#include "SceneRenderPipeline.h"

namespace LTSE::Graphics
{

    using namespace math;
    namespace fs = std::filesystem;

    struct VisualHelperMeshRendererCreateInfo
    {
        float LineWidth                        = 1.0f;
        Ref<LTSE::Graphics::Internal::sVkRenderPassObject>  RenderPass = nullptr;
    };

    class VisualHelperMeshRenderer : public LTSE::Core::SceneRenderPipeline<SimpleVertexData>
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
        VisualHelperMeshRenderer( GraphicContext &a_GraphicContext, VisualHelperMeshRendererCreateInfo a_CreateInfo );

        ~VisualHelperMeshRenderer() = default;

        void Render( math::mat4 a_Model, math::mat4 a_View, math::mat4 a_Projection, math::vec3 a_Color, Ref<Buffer> a_VertexBuffer,
                     Ref<Buffer> a_IndexBuffer, RenderContext &aRenderContext );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange> GetPushConstantLayout();
    };

} // namespace LTSE::Graphics
