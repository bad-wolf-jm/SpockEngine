#include "EffectProcessor.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

namespace SE::Core
{

    std::vector<Ref<DescriptorSetLayout>> EffectProcessor::GetDescriptorSetLayout() { return { PipelineLayout }; }

    std::vector<sPushConstantRange> EffectProcessor::GetPushConstantLayout() { return {}; };

    EffectProcessor::EffectProcessor( Ref<VkGraphicContext> aGraphicContext, ARenderContext &aRenderContext,
                                      EffectProcessorCreateInfo aCreateInfo )
        : SceneRenderPipeline<EmptyVertexData>( aGraphicContext )
        , Spec{ aCreateInfo }
    {
        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.IsTwoSided     = true;
        lCreateInfo.LineWidth      = 1.0f;
        lCreateInfo.VertexShader   = "Shaders\\fxaa.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\fxaa.frag.spv";
        lCreateInfo.RenderPass     = aRenderContext.GetRenderPass();

        DescriptorSetLayoutCreateInfo lPipelineLayoutCI{};
        lPipelineLayoutCI.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } } };
        PipelineLayout = New<DescriptorSetLayout>( mGraphicContext, lPipelineLayoutCI );

        Initialize( lCreateInfo );

        mCameraBuffer =
            New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( sCameraViewUniforms ) );
        mCameraDescriptors = New<DescriptorSet>( aGraphicContext, PipelineLayout );
        mCameraDescriptors->Write( mCameraBuffer, false, 0, sizeof( sCameraViewUniforms ), 0 );
    }

    void EffectProcessor::Render( math::mat4 aProjection, math::mat4 aView, ARenderContext &aRenderContext )
    {
        sCameraViewUniforms l_View{ aView, aProjection };

        mCameraBuffer->Write( l_View );
        aRenderContext.Bind( Pipeline );
        aRenderContext.Bind( mCameraDescriptors, 0, -1 );
        aRenderContext.ResetBuffers();
        aRenderContext.Draw( 6, 0, 0, 1, 0 );
    }

} // namespace SE::Core