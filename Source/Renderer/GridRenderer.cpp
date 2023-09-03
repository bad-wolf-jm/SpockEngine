#include "GridRenderer.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

namespace SE::Core
{
    using namespace SE::Graphics;

    CoordinateGridRenderer::CoordinateGridRenderer( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderContext> aRenderContext )
        : mGraphicContext( aGraphicContext )
    {
        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::NONE );
        mPipeline->SetDepthParameters( true, true, eDepthCompareOperation::LESS_OR_EQUAL );

        auto lVertexShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, "coordinate_grid_vertex_shader" );
        lVertexShader->AddCode( "#define __GLSL__" );
        lVertexShader->AddCode( "#define VULKAN_SEMANTICS" );
        lVertexShader->AddCode( "#define COORDINATE_GRID_VERTEX_SHADER" );
        lVertexShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\Definitions.hpp" );
        lVertexShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\CoordinateGrid.hpp" );
        lVertexShader->Compile();

        auto lFragmentShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::FRAGMENT, 450, "coordinate_grid_fragment_shader" );
        lFragmentShader->AddCode( "#define __GLSL__" );
        lFragmentShader->AddCode( "#define VULKAN_SEMANTICS" );
        lFragmentShader->AddCode( "#define COORDINATE_GRID_FRAGMENT_SHADER" );
        lFragmentShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\Common\\Definitions.hpp" );
        lFragmentShader->AddFile( "D:\\Work\\Git\\SpockEngine\\Shaders\\Source\\CoordinateGrid.hpp" );
        lFragmentShader->Compile();

        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, lFragmentShader, "main" );
        mPipeline->AddPushConstantRange( { eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 );

        mPipelineLayout = CreateDescriptorSetLayout( aGraphicContext );
        mPipelineLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        mPipelineLayout->Build();

        mPipeline->AddDescriptorSet( mPipelineLayout );

        mPipeline->Build();

        mCameraBuffer =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( CameraViewUniforms ) );

        mCameraDescriptors = mPipelineLayout->Allocate();
        mCameraDescriptors->Write( mCameraBuffer, false, 0, sizeof( CameraViewUniforms ), 0 );
    }

    void CoordinateGridRenderer::Render( math::mat4 aProjection, math::mat4 aView, ref_t<IRenderContext> aRenderContext )
    {
        CameraViewUniforms lView{ aView, aProjection };

        mCameraBuffer->Write( lView );

        aRenderContext->Bind( mPipeline );
        aRenderContext->Bind( mCameraDescriptors, 0, -1 );
        aRenderContext->ResetBuffers();
        aRenderContext->Draw( 6, 0, 0, 1, 0 );
    }

} // namespace SE::Core