#include "VisualHelperLineRenderer.h"

#include "Core/Resource.h"

#include "Scene/VertexData.h"

using namespace SE::Core;

namespace SE::Graphics
{

    std::vector<Ref<IDescriptorSetLayout>> VisualHelperLineRenderer::GetDescriptorSetLayout() { return {}; }

    std::vector<sPushConstantRange> VisualHelperLineRenderer::GetPushConstantLayout()
    {
        return { { { eShaderStageTypeFlags::VERTEX, eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( CameraViewUniforms ) } };
    };

    VisualHelperLineRenderer::VisualHelperLineRenderer( Ref<IGraphicContext>               a_GraphicContext,
                                                        VisualHelperLineRendererCreateInfo a_CreateInfo )
        : SceneRenderPipeline<PositionData>( a_GraphicContext )
        , Spec{ a_CreateInfo }
    {
        SceneRenderPipelineCreateInfo l_CreateInfo{};
        l_CreateInfo.Topology       = ePrimitiveTopology::LINES;
        l_CreateInfo.LineWidth      = a_CreateInfo.LineWidth;
        l_CreateInfo.VertexShader   = "Shaders/WireframeShader.vert";
        l_CreateInfo.FragmentShader = "Shaders/WireframeShader.frag";
        l_CreateInfo.RenderPass     = a_CreateInfo.RenderPass;

        Initialize( l_CreateInfo );
    }

    void VisualHelperLineRenderer::Render( math::mat4 aModel, math::mat4 aView, math::mat4 aProjection, math::vec3 aColor,
                                           Ref<VkGpuBuffer> aVertexBuffer, Ref<VkGpuBuffer> aIndexBuffer,
                                           Ref<iRenderContext> aRenderContext )
    {
        CameraViewUniforms lView{ aModel, aView, aProjection, math::vec4( aColor, 1.0f ) };

        aRenderContext->Bind( Pipeline );
        aRenderContext->PushConstants( { eShaderStageTypeFlags::VERTEX, eShaderStageTypeFlags::FRAGMENT }, 0, lView );
        aRenderContext->Bind( aVertexBuffer, aIndexBuffer, 0 );
        aRenderContext->Draw( aIndexBuffer->SizeAs<uint32_t>(), 0, 0, 1, 0 );
    }

} // namespace SE::Graphics
