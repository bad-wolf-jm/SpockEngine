#include "MeshRenderer.h"

#include <chrono>

#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace SE::Core
{

    using namespace math;

    Ref<IDescriptorSetLayout> MeshRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER,
                                { eShaderStageTypeFlags::VERTEX, eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> MeshRenderer::GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        lNewLayout->AddBinding( 0, eDescriptorType::STORAGE_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> MeshRenderer::GetNodeSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        lNewLayout->Build();

        return lNewLayout;
    }

    // std::vector<Ref<IDescriptorSetLayout>> MeshRenderer::GetDescriptorSetLayout()
    // {
    //     return { CameraSetLayout, TextureSetLayout, NodeSetLayout };
    // }

    // std::vector<sPushConstantRange> MeshRenderer::GetPushConstantLayout()
    // {
    //     return { { { eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) } };
    // };

    MeshRenderer::MeshRenderer( Ref<IGraphicContext> aGraphicContext, MeshRendererCreateInfo const &aCreateInfo )
        : mGraphicContext{ aGraphicContext }
        , Spec{ aCreateInfo }
    {

        mPipeline = CreateGraphicsPipeline( mGraphicContext, aCreateInfo.RenderPass, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::BACK );
        mPipeline->SetLineWidth( Spec.LineWidth );
        mPipeline->SetDepthParameters(true, true, eDepthCompareOperation::LESS_OR_EQUAL);
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, GetResourcePath( Spec.VertexShader ), "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, GetResourcePath( Spec.FragmentShader ), "main" );
        mPipeline->AddPushConstantRange( { eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) );
        mPipeline->AddInput( "Position", eBufferDataType::VEC3, 0, 0 );
        mPipeline->AddInput( "Normal", eBufferDataType::VEC3, 0, 1 );
        mPipeline->AddInput( "TexCoord_0", eBufferDataType::VEC2, 0, 2 );
        mPipeline->AddInput( "Bones", eBufferDataType::VEC4, 0, 3 );
        mPipeline->AddInput( "Weights", eBufferDataType::VEC4, 0, 4 );

        CameraSetLayout  = GetCameraSetLayout( mGraphicContext );
        TextureSetLayout = GetTextureSetLayout( mGraphicContext );
        NodeSetLayout    = GetNodeSetLayout( mGraphicContext );

        mPipeline->AddDescriptorSet( CameraSetLayout );
        mPipeline->AddDescriptorSet( TextureSetLayout );
        mPipeline->AddDescriptorSet( NodeSetLayout );

        mPipeline->Build();
    }

} // namespace SE::Core