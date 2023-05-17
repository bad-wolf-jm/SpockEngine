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
        lNewLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> MeshRenderer::GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
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
        : Spec{ aCreateInfo }
    {

        mPipeline = CreateGraphicsPipeline( mGraphicContext, aCreateInfo.RenderPass, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::BACK );
        mPipeline->SetLineWidth( Spec.LineWidth );
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, Spec.VertexShader, "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, Spec.FragmentShader, "main" );
        mPipeline->AddPushConstantRange( { eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) );

        CameraSetLayout  = GetCameraSetLayout( mGraphicContext );
        TextureSetLayout = GetTextureSetLayout( mGraphicContext );
        NodeSetLayout    = GetNodeSetLayout( mGraphicContext );

        mPipeline->AddDescriptorSet( CameraSetLayout );
        mPipeline->AddDescriptorSet( TextureSetLayout );
        mPipeline->AddDescriptorSet( NodeSetLayout );

        mPipeline->Build();
    }

} // namespace SE::Core