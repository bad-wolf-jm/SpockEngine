#include "VisualHelperLineRenderer.h"

#include "Core/Resource.h"

#include "Scene/VertexData.h"

using namespace LTSE::Core;

namespace LTSE::Graphics
{

    std::vector<Ref<DescriptorSetLayout>> VisualHelperLineRenderer::GetDescriptorSetLayout() { return {}; }

    std::vector<sPushConstantRange> VisualHelperLineRenderer::GetPushConstantLayout()
    {
        return { { { Internal::eShaderStageTypeFlags::VERTEX, Internal::eShaderStageTypeFlags::FRAGMENT }, 0,
            sizeof( CameraViewUniforms ) } };
    };

    VisualHelperLineRenderer::VisualHelperLineRenderer(
        GraphicContext &a_GraphicContext, VisualHelperLineRendererCreateInfo a_CreateInfo )
        : SceneRenderPipeline<PositionData>( a_GraphicContext )
        , Spec{ a_CreateInfo }
    {
        SceneRenderPipelineCreateInfo l_CreateInfo{};
        l_CreateInfo.Topology       = ePrimitiveTopology::LINES;
        l_CreateInfo.LineWidth      = a_CreateInfo.LineWidth;
        l_CreateInfo.VertexShader   = "Shaders/WireframeShader.vert.spv";
        l_CreateInfo.FragmentShader = "Shaders/WireframeShader.frag.spv";
        l_CreateInfo.RenderPass     = a_CreateInfo.RenderPass;

        Initialize( l_CreateInfo );
    }

    void VisualHelperLineRenderer::Render( math::mat4 a_Model, math::mat4 a_View, math::mat4 a_Projection, math::vec3 a_Color,
        Ref<Buffer> a_VertexBuffer, Ref<Buffer> a_IndexBuffer, RenderContext &aRenderContext )
    {
        CameraViewUniforms l_View{ a_Model, a_View, a_Projection, math::vec4( a_Color, 1.0f ) };

        aRenderContext.Bind( Pipeline );
        aRenderContext.PushConstants(
            { Internal::eShaderStageTypeFlags::VERTEX, Internal::eShaderStageTypeFlags::FRAGMENT }, 0, l_View );
        aRenderContext.Bind( a_VertexBuffer, a_IndexBuffer, 0 );
        aRenderContext.Draw( a_IndexBuffer->SizeAs<uint32_t>(), 0, 0, 1, 0 );
    }

    void VisualHelperLineRenderer::Render( math::mat4 a_Model, math::mat4 a_View, math::mat4 a_Projection, math::vec3 a_Color,
        Ref<Buffer> a_VertexBuffer, Ref<Buffer> a_IndexBuffer, ARenderContext &aRenderContext )
    {
        CameraViewUniforms l_View{ a_Model, a_View, a_Projection, math::vec4( a_Color, 1.0f ) };

        aRenderContext.Bind( Pipeline );
        aRenderContext.PushConstants(
            { Internal::eShaderStageTypeFlags::VERTEX, Internal::eShaderStageTypeFlags::FRAGMENT }, 0, l_View );
        aRenderContext.Bind( a_VertexBuffer, a_IndexBuffer, 0 );
        aRenderContext.Draw( a_IndexBuffer->SizeAs<uint32_t>(), 0, 0, 1, 0 );
    }

} // namespace LTSE::Graphics
