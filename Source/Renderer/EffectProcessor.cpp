#include "EffectProcessor.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

namespace SE::Core
{
    using namespace SE::Graphics;

    EffectProcessor::EffectProcessor( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderContext> aRenderContext,
                                      EffectProcessorCreateInfo aCreateInfo )
        : mGraphicContext{ aGraphicContext }
        , Spec{ aCreateInfo }
    {
        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::NONE );
        mPipeline->SetLineWidth( 1.0f );
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, Spec.mVertexShader, "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, Spec.mFragmentShader, "main" );

        PipelineLayout = CreateDescriptorSetLayout( mGraphicContext );
        PipelineLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        PipelineLayout->Build();
        mPipeline->AddDescriptorSet( PipelineLayout );

        mPipeline->Build();

        mTextures = PipelineLayout->Allocate();
    }

    void EffectProcessor::Render( ref_t<ISampler2D> aImageSampler, ref_t<IRenderContext> aRenderContext )
    {
        mTextures->Write( aImageSampler, 0 );

        aRenderContext->Bind( mPipeline );
        aRenderContext->Bind( mTextures, 0, -1 );
        aRenderContext->ResetBuffers();
        aRenderContext->Draw( 6, 0, 0, 1, 0 );
    }

} // namespace SE::Core