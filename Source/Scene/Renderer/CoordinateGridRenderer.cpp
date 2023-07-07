#include "CoordinateGridRenderer.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

#include "Shaders/Embedded/gCoordinateGridFragmentShader.h"
#include "Shaders/Embedded/gCoordinateGridVertexShader.h"

namespace SE::Core
{
    using namespace SE::Graphics;

    CoordinateGridRenderer::CoordinateGridRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext )
        : mGraphicContext( aGraphicContext )
    {
        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::NONE );
        mPipeline->SetDepthParameters( true, true, eDepthCompareOperation::LESS_OR_EQUAL );


        fs::path lShaderPath = "C:\\GitLab\\SpockEngine\\Resources\\Shaders\\Cache";
        auto     lVertexShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, "coordinate_grid_renderer_vertex_shader", lShaderPath );
        lVertexShader->AddCode( SE::Private::Shaders::gCoordinateGridVertexShader_data );
        lVertexShader->Compile();

        auto lFragmentShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::FRAGMENT, 450, "coordinate_grid_renderer_fragment_shader", lShaderPath );
        lFragmentShader->AddCode( SE::Private::Shaders::gCoordinateGridFragmentShader_data );
        lFragmentShader->Compile();


        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, lFragmentShader, "main" );
        mPipeline->AddPushConstantRange( { eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 );

        PipelineLayout = CreateDescriptorSetLayout( aGraphicContext );
        PipelineLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        PipelineLayout->Build();

        mPipeline->AddDescriptorSet( PipelineLayout );

        mPipeline->Build();

        mCameraBuffer =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( CameraViewUniforms ) );

        mCameraDescriptors = PipelineLayout->Allocate();
        mCameraDescriptors->Write( mCameraBuffer, false, 0, sizeof( CameraViewUniforms ), 0 );
    }

    void CoordinateGridRenderer::Render( math::mat4 aProjection, math::mat4 aView, Ref<IRenderContext> aRenderContext )
    {
        CameraViewUniforms lView{ aView, aProjection };

        mCameraBuffer->Write( lView );

        aRenderContext->Bind( mPipeline );
        aRenderContext->Bind( mCameraDescriptors, 0, -1 );
        aRenderContext->ResetBuffers();
        aRenderContext->Draw( 6, 0, 0, 1, 0 );
    }

} // namespace SE::Core