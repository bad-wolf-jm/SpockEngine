#include "CoordinateGridRenderer.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

namespace SE::Core
{
    using namespace SE::Graphics;
    // std::vector<Ref<DescriptorSetLayout>> CoordinateGridRenderer::GetDescriptorSetLayout() { return { PipelineLayout }; }

    // std::vector<sPushConstantRange> CoordinateGridRenderer::GetPushConstantLayout() { return {}; };

    CoordinateGridRenderer::CoordinateGridRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext )
        : mGraphicContext( aGraphicContext )
    // , Spec{ aCreateInfo }
    {
        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::NONE );
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, GetResourcePath( "Shaders\\coordinategrid.vert.spv" ), "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, GetResourcePath( "Shaders\\coordinategrid.frag.spv" ), "main" );
        mPipeline->AddPushConstantRange( { eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 );

        PipelineLayout = CreateDescriptorSetLayout( aGraphicContext );
        PipelineLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        PipelineLayout->Build();

        mPipeline->AddDescriptorSet( PipelineLayout );

        mPipeline->Build();

        // SceneRenderPipelineCreateInfo lCreateInfo{};
        // lCreateInfo.IsTwoSided     = true;
        // lCreateInfo.LineWidth      = 1.0f;
        // lCreateInfo.VertexShader   = "Shaders\\coordinategrid.vert.spv";
        // lCreateInfo.FragmentShader = "Shaders\\coordinategrid.frag.spv";
        // lCreateInfo.RenderPass     = aRenderContext.GetRenderPass();
        // DescriptorSetLayoutCreateInfo lPipelineLayoutCI{};
        // lPipelineLayoutCI.Bindings = {
        //     DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } } };
        // PipelineLayout = New<DescriptorSetLayout>( mGraphicContext, lPipelineLayoutCI );
        // Initialize( lCreateInfo );

        // New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( CameraViewUniforms ) );
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