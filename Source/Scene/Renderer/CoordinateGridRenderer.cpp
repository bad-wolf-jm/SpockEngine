#include "CoordinateGridRenderer.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

namespace SE::Core
{

    std::vector<Ref<DescriptorSetLayout>> CoordinateGridRenderer::GetDescriptorSetLayout() { return { PipelineLayout }; }

    std::vector<sPushConstantRange> CoordinateGridRenderer::GetPushConstantLayout() { return {}; };

    CoordinateGridRenderer::CoordinateGridRenderer( Ref<VkGraphicContext> a_GraphicContext, ARenderContext &aRenderContext,
                                                    CoordinateGridRendererCreateInfo aCreateInfo )
        : SceneRenderPipeline<EmptyVertexData>( a_GraphicContext )
        , Spec{ aCreateInfo }
    {
        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.IsTwoSided     = true;
        lCreateInfo.LineWidth      = 1.0f;
        lCreateInfo.VertexShader   = "Shaders\\coordinategrid.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\coordinategrid.frag.spv";
        lCreateInfo.RenderPass     = aRenderContext.GetRenderPass();

        DescriptorSetLayoutCreateInfo lPipelineLayoutCI{};
        lPipelineLayoutCI.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } } };
        PipelineLayout = New<DescriptorSetLayout>( mGraphicContext, lPipelineLayoutCI );

        Initialize( lCreateInfo );

        mCameraBuffer =
            New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( CameraViewUniforms ) );
        mCameraDescriptors = New<DescriptorSet>( a_GraphicContext, PipelineLayout );
        mCameraDescriptors->Write( mCameraBuffer, false, 0, sizeof( CameraViewUniforms ), 0 );
    }

    void CoordinateGridRenderer::Render( math::mat4 a_Projection, math::mat4 a_View, ARenderContext &aRenderContext )
    {
        CameraViewUniforms l_View{ a_View, a_Projection };

        mCameraBuffer->Write( l_View );
        aRenderContext.Bind( Pipeline );
        aRenderContext.Bind( mCameraDescriptors, 0, -1 );
        aRenderContext.ResetBuffers();
        aRenderContext.Draw( 6, 0, 0, 1, 0 );
    }

} // namespace SE::Core