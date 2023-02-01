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
        lCreateInfo.VertexShader   = aCreateInfo.mVertexShader;
        lCreateInfo.FragmentShader = aCreateInfo.mFragmentShader;
        lCreateInfo.RenderPass     = aRenderContext.GetRenderPass();

        DescriptorSetLayoutCreateInfo lPipelineLayoutCI{};
        lPipelineLayoutCI.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } } };
        PipelineLayout = New<DescriptorSetLayout>( mGraphicContext, lPipelineLayoutCI );

        Initialize( lCreateInfo );
    }

    void EffectProcessor::Render( ARenderContext &aRenderContext )
    {
        aRenderContext.Bind( Pipeline );
        aRenderContext.ResetBuffers();
        aRenderContext.Draw( 6, 0, 0, 1, 0 );
    }

} // namespace SE::Core