#include "EffectProcessor.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

namespace SE::Core
{

    // std::vector<Ref<DescriptorSetLayout>> EffectProcessor::GetDescriptorSetLayout() { return { PipelineLayout }; }
    // std::vector<sPushConstantRange> EffectProcessor::GetPushConstantLayout() { return {}; };

    EffectProcessor::EffectProcessor( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext,
                                      EffectProcessorCreateInfo aCreateInfo )
        : Spec{ aCreateInfo }
    {
        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::BACK );
        mPipeline->SetLineWidth( Spec.LineWidth );
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, Spec.mVertexShader, "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, Spec.mFragmentShader, "main" );

        PipelineLayout = CrerateDescriptorSetLayout( mGraphicContext );
        PipelineLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        PipelineLayout->Build();
        mPipeline->AddDescriptorSet( PipelineLayout );

        mPipeline->Build();

        mTextures = PipelineLayout->Allocate();
        // SceneRenderPipelineCreateInfo lCreateInfo{};
        // lCreateInfo.IsTwoSided     = true;
        // lCreateInfo.LineWidth      = 1.0f;
        // lCreateInfo.VertexShader   = aCreateInfo.mVertexShader;
        // lCreateInfo.FragmentShader = aCreateInfo.mFragmentShader;
        // lCreateInfo.DepthTest      = false;
        // lCreateInfo.DepthWrite     = false;
        // lCreateInfo.RenderPass     = aRenderContext->GetRenderPass();

        // DescriptorSetLayoutCreateInfo lPipelineLayoutCI{};
        // lPipelineLayoutCI.Bindings = {
        //     DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };
        // PipelineLayout = New<DescriptorSetLayout>( mGraphicContext, lPipelineLayoutCI );

        // Initialize( lCreateInfo );
    }

    void EffectProcessor::Render( Ref<ISampler2D> aImageSampler, Ref<IRenderContext> aRenderContext )
    {
        mTextures->Write( aImageSampler, 0 );

        aRenderContext->Bind( Pipeline );
        aRenderContext->Bind( mTextures, 0, -1 );
        aRenderContext->ResetBuffers();
        aRenderContext->Draw( 6, 0, 0, 1, 0 );
    }

} // namespace SE::Core