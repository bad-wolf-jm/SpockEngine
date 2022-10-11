#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Types.h"
#include "Core/Memory.h"

#include "Developer/GraphicContext/Buffer.h"
#include "Developer/GraphicContext/DescriptorSet.h"
#include "Developer/GraphicContext/RenderContext.h"

#include "Developer/GraphicContext/GraphicContext.h"
#include "Developer/GraphicContext/GraphicsPipeline.h"
#include "Developer/Core/Vulkan/VkRenderPass.h"

#include "Developer/Scene/ParticleData.h"
#include "Developer/Scene/VertexData.h"

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
