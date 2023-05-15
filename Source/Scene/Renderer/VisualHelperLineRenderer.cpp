#include "VisualHelperLineRenderer.h"

#include "Core/Resource.h"

#include "Scene/VertexData.h"

using namespace SE::Core;

namespace SE::Graphics
{

    std::vector<Ref<DescriptorSetLayout>> VisualHelperLineRenderer::GetDescriptorSetLayout() { return {}; }

    std::vector<sPushConstantRange> VisualHelperLineRenderer::GetPushConstantLayout()
    {
        return { { { eShaderStageTypeFlags::VERTEX, eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( CameraViewUniforms ) } };
    };

    VisualHelperLineRenderer::VisualHelperLineRenderer( Ref<IGraphicContext>              a_GraphicContext,
                                                        VisualHelperLineRendererCreateInfo a_CreateInfo )
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
                                           Ref<VkGpuBuffer> a_VertexBuffer, Ref<VkGpuBuffer> a_IndexBuffer,
                                           VkRenderContext &aRenderContext )
    {
        CameraViewUniforms l_View{ a_Model, a_View, a_Projection, math::vec4( a_Color, 1.0f ) };

        aRenderContext.Bind( Pipeline );
        aRenderContext.PushConstants( { eShaderStageTypeFlags::VERTEX, eShaderStageTypeFlags::FRAGMENT }, 0, l_View );
        aRenderContext.Bind( a_VertexBuffer, a_IndexBuffer, 0 );
        aRenderContext.Draw( a_IndexBuffer->SizeAs<uint32_t>(), 0, 0, 1, 0 );
    }

} // namespace SE::Graphics
