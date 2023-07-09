#include "ParticleSystemRenderer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

using namespace SE::Core;

namespace SE::Graphics
{
    ParticleSystemRenderer::ParticleSystemRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext,
                                                    ParticleRendererCreateInfo aCreateInfo )
        : mGraphicContext{ aGraphicContext }
        , Spec{ aCreateInfo }
    {
        mCameraBuffer =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, false, true, true, sizeof( CameraViewUniforms ) );

        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::NONE );
        mPipeline->SetDepthParameters( true, true, eDepthCompareOperation::LESS_OR_EQUAL );
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, aCreateInfo.VertexShader , "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, aCreateInfo.FragmentShader , "main" );
        mPipeline->AddPushConstantRange( { eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 );

        auto lDescriptorSet = CreateDescriptorSetLayout( aGraphicContext );
        lDescriptorSet->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        lDescriptorSet->Build();

        mPipeline->AddDescriptorSet( lDescriptorSet );
        mPipeline->Build();

        mCameraDescriptors = lDescriptorSet->Allocate();
        mCameraDescriptors->Write( mCameraBuffer, false, 0, sizeof( CameraViewUniforms ), 0 );

        std::vector<math::vec3> lVertexBufferData = {
            { -.5f, -.5f, 0.0f }, { -.5f, .5f, 0.0f }, { .5f, .5f, 0.0f }, { .5f, -.5f, 0.0f } };
        std::vector<uint32_t> lIndexBufferData = { 0, 2, 1, 0, 3, 2 };

        mParticleVertices = CreateBuffer( mGraphicContext, lVertexBufferData, eBufferType::VERTEX_BUFFER, false, false, false, true );
        mParticleIndices  = CreateBuffer( mGraphicContext, lIndexBufferData, eBufferType::INDEX_BUFFER, false, false, false, true );
    }

    void ParticleSystemRenderer::Render( math::mat4 aProjection, math::mat4 aView, Ref<IRenderContext> aRenderContext,
                                         ParticleData &aParticleData )
    {
        if( aParticleData.Particles == nullptr ) return;

        CameraViewUniforms l_View{ aParticleData.Model, aView, aProjection, aParticleData.ParticleSize };

        mCameraBuffer->Write( l_View );
        aRenderContext->Bind( mPipeline );
        aRenderContext->Bind( mCameraDescriptors, 0, -1 );
        aRenderContext->Bind( mParticleVertices, mParticleIndices, 0 );
        aRenderContext->Bind( aParticleData.Particles, 1 );
        aRenderContext->Draw( 6, 0, 0, aParticleData.ParticleCount, 0 );
    }

} // namespace SE::Graphics
