#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"


#include "Core/GraphicContext//Buffer.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//RenderContext.h"

#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//Texture2D.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/Vulkan/VkRenderPass.h"


#include "Scene/VertexData.h"

#include "SceneRenderPipeline.h"

namespace LTSE::Graphics
{
    using namespace math;
    using namespace LTSE::Core;
    namespace fs = std::filesystem;

    struct VisualHelperLineRendererCreateInfo
    {
        float LineWidth                                              = 1.0f;
        Ref<LTSE::Graphics::Internal::sVkRenderPassObject> RenderPass = nullptr;
    };

    class VisualHelperLineRenderer : public SceneRenderPipeline<PositionData>
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
        VisualHelperLineRenderer( GraphicContext &a_GraphicContext, VisualHelperLineRendererCreateInfo a_CreateInfo );
        ~VisualHelperLineRenderer() = default;

        void Render( math::mat4 a_Model, math::mat4 a_View, math::mat4 a_Projection, math::vec3 a_Color, Ref<Buffer> a_VertexBuffer,
                     Ref<Buffer> a_IndexBuffer, RenderContext &aRenderContext );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange> GetPushConstantLayout();
    };

} // namespace LTSE::Graphics
