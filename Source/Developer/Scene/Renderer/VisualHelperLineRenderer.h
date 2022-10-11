#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"


#include "Developer/GraphicContext/Buffer.h"
#include "Developer/GraphicContext/DescriptorSet.h"
#include "Developer/GraphicContext/RenderContext.h"

#include "Developer/GraphicContext/GraphicContext.h"
#include "Developer/GraphicContext/Texture2D.h"
#include "Developer/GraphicContext/GraphicsPipeline.h"
#include "Developer/Core/Vulkan/VkRenderPass.h"


#include "Developer/Scene/VertexData.h"

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
