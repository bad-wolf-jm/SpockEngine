#include "CoordinateGridRenderer.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

namespace LTSE::Core
{

    std::vector<Ref<DescriptorSetLayout>> CoordinateGridRenderer::GetDescriptorSetLayout() { return { PipelineLayout }; }

    std::vector<sPushConstantRange> CoordinateGridRenderer::GetPushConstantLayout() { return {}; };

    CoordinateGridRenderer::CoordinateGridRenderer(
        GraphicContext &a_GraphicContext, RenderContext &a_RenderContext, CoordinateGridRendererCreateInfo a_CreateInfo )
        : SceneRenderPipeline<EmptyVertexData>( a_GraphicContext )
        , Spec{ a_CreateInfo }
    {
        SceneRenderPipelineCreateInfo l_CreateInfo{};
        l_CreateInfo.IsTwoSided     = true;
        l_CreateInfo.LineWidth      = 1.0f;
        l_CreateInfo.VertexShader   = "Shaders\\coordinategrid.vert.spv";
        l_CreateInfo.FragmentShader = "Shaders\\coordinategrid.frag.spv";
        l_CreateInfo.RenderPass     = a_CreateInfo.RenderPass;

        DescriptorSetLayoutCreateInfo l_PipelineLayoutCI{};
        l_PipelineLayoutCI.Bindings = {
            DescriptorBindingInfo{ 0, Internal::eDescriptorType::UNIFORM_BUFFER, { Internal::eShaderStageTypeFlags::VERTEX } } };
        PipelineLayout = New<DescriptorSetLayout>( mGraphicContext, l_PipelineLayoutCI );

        Initialize( l_CreateInfo );

        mCameraBuffer =
            New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( CameraViewUniforms ) );
        mCameraDescriptors = New<DescriptorSet>( a_GraphicContext, PipelineLayout );
        mCameraDescriptors->Write( mCameraBuffer, false, 0, sizeof( CameraViewUniforms ), 0 );
    }

    void CoordinateGridRenderer::Render( math::mat4 a_Projection, math::mat4 a_View, RenderContext &aRenderContext )
    {
        CameraViewUniforms l_View{ a_View, a_Projection };

        mCameraBuffer->Write( l_View );
        aRenderContext.Bind( Pipeline );
        aRenderContext.Bind( mCameraDescriptors, 0, -1 );
        aRenderContext.ResetBuffers();
        aRenderContext.Draw( 6, 0, 0, 1, 0 );
    }

} // namespace LTSE::Core