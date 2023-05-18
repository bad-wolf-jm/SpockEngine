#include "DeferredLightingRenderer.h"

#include <chrono>

#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

namespace SE::Core
{
    using namespace math;
    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );

        DescriptorSetLayoutCreateInfo lTextureBindLayout{};
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 2, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 3, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetDirectionalShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetSpotlightShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetPointLightShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        lNewLayout->Build();

        return lNewLayout;
    }

    DeferredLightingRenderer::DeferredLightingRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext )
        : mGraphicContext( aGraphicContext )
    {

        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::NONE );
        mPipeline->SetDepthParameters( false, false, eDepthCompareOperation::ALWAYS );
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, GetResourcePath( "Shaders/Deferred/DeferredLightingMSAA.vert.spv" ),
                              "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, GetResourcePath( "Shaders/Deferred/DeferredLightingMSAA.frag.spv" ),
                              "main" );

        CameraSetLayout            = GetCameraSetLayout( mGraphicContext );
        TextureSetLayout           = GetTextureSetLayout( mGraphicContext );
        DirectionalShadowSetLayout = GetDirectionalShadowSetLayout( mGraphicContext );
        SpotlightShadowSetLayout   = GetSpotlightShadowSetLayout( mGraphicContext );
        PointLightShadowSetLayout  = GetPointLightShadowSetLayout( mGraphicContext );

        mPipeline->AddDescriptorSet( CameraSetLayout );
        mPipeline->AddDescriptorSet( TextureSetLayout );
        mPipeline->AddDescriptorSet( DirectionalShadowSetLayout );
        mPipeline->AddDescriptorSet( SpotlightShadowSetLayout );
        mPipeline->AddDescriptorSet( PointLightShadowSetLayout );

        mPipeline->Build();
    }

} // namespace SE::Core