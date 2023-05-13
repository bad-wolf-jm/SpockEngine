#include "EffectProcessor.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

namespace SE::Core
{

    std::vector<Ref<DescriptorSetLayout>> EffectProcessor::GetDescriptorSetLayout() { return { PipelineLayout }; }

    std::vector<sPushConstantRange> EffectProcessor::GetPushConstantLayout() { return {}; };

    EffectProcessor::EffectProcessor( Ref<VkGraphicContext> aGraphicContext, VkRenderContext &aRenderContext,
                                      EffectProcessorCreateInfo aCreateInfo )
        : SceneRenderPipeline<EmptyVertexData>( aGraphicContext )
        , Spec{ aCreateInfo }
    {
        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.IsTwoSided     = true;
        lCreateInfo.LineWidth      = 1.0f;
        lCreateInfo.VertexShader   = aCreateInfo.mVertexShader;
        lCreateInfo.FragmentShader = aCreateInfo.mFragmentShader;
        lCreateInfo.DepthTest      = false;
        lCreateInfo.DepthWrite     = false;
        lCreateInfo.RenderPass     = aRenderContext.GetRenderPass();

        DescriptorSetLayoutCreateInfo lPipelineLayoutCI{};
        lPipelineLayoutCI.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };
        PipelineLayout = New<DescriptorSetLayout>( mGraphicContext, lPipelineLayoutCI );
        mTextures      = New<DescriptorSet>( mGraphicContext, PipelineLayout );

        Initialize( lCreateInfo );
    }

    void EffectProcessor::Render( Ref<Graphics::VkSampler2D> aImageSampler, VkRenderContext &aRenderContext )
    {
        mTextures->Write(aImageSampler, 0);

        aRenderContext.Bind( Pipeline );
        aRenderContext.Bind( mTextures, 0, -1  );
        aRenderContext.ResetBuffers();
        aRenderContext.Draw( 6, 0, 0, 1, 0 );
    }

} // namespace SE::Core