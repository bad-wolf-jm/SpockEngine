#include "VisualHelperMeshRenderer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

using namespace SE::Core;

namespace SE::Graphics
{

    std::vector<Ref<IDescriptorSetLayout>> VisualHelperMeshRenderer::GetDescriptorSetLayout() { return {}; }

    std::vector<sPushConstantRange> VisualHelperMeshRenderer::GetPushConstantLayout()
    {
        return { { { eShaderStageTypeFlags::VERTEX, eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( CameraViewUniforms ) } };
    };

    VisualHelperMeshRenderer::VisualHelperMeshRenderer( Ref<IGraphicContext>               aGraphicContext,
                                                        VisualHelperMeshRendererCreateInfo aCreateInfo )
        : SceneRenderPipeline<SimpleVertexData>( aGraphicContext )
        , Spec{ aCreateInfo }
    {
        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.IsTwoSided     = true;
        lCreateInfo.LineWidth      = aCreateInfo.LineWidth;
        lCreateInfo.VertexShader   = "Shaders/Unlit.vert";
        lCreateInfo.FragmentShader = "Shaders/Unlit.frag";
        lCreateInfo.RenderPass     = aCreateInfo.RenderPass;

        Initialize( lCreateInfo );
    }

    void VisualHelperMeshRenderer::Render( math::mat4 aModel, math::mat4 aView, math::mat4 aProjection, math::vec3 aColor,
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
