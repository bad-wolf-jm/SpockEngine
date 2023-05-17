#include "DeferredLightingRenderer.h"

#include <chrono>

// #include "Graphics/Vulkan/VkPipeline.h"
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
        // lTextureBindLayout.Bindings = {
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

        // DescriptorSetLayoutCreateInfo lShadowMapLayout{};
        // lShadowMapLayout.Bindings = {
        //     DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetSpotlightShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        // DescriptorSetLayoutCreateInfo lShadowMapLayout{};
        // lShadowMapLayout.Bindings = {
        //     DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetPointLightShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        // DescriptorSetLayoutCreateInfo lShadowMapLayout{};
        // lShadowMapLayout.Bindings = {
        //     DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

        lNewLayout->Build();

        return lNewLayout;
    }

    // std::vector<Ref<IDescriptorSetLayout>> DeferredLightingRenderer::GetDescriptorSetLayout()
    // {
    //     return { CameraSetLayout, TextureSetLayout, DirectionalShadowSetLayout, SpotlightShadowSetLayout, PointLightShadowSetLayout
    //     };
    // }

    // std::vector<sPushConstantRange> DeferredLightingRenderer::GetPushConstantLayout()
    // {
    //     return { { { eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) } };
    // };

    DeferredLightingRenderer::DeferredLightingRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext )
        : mGraphicContext( aGraphicContext )
    {

        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::NONE );
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
        // auto &lCameraDescriptorSet = mPipeline->AddDescriptorSet();
        // lCameraDescriptorSet.Add( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        // lCameraDescriptorSet.Add( 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );

        // auto &lTextureDescriptorSet = mPipeline->AddDescriptorSet();
        // lTextureDescriptorSet.Add( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        // lTextureDescriptorSet.Add( 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        // lTextureDescriptorSet.Add( 2, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        // lTextureDescriptorSet.Add( 3, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        // auto &lDirectionalShadowDescriptorSet = mPipeline->AddDescriptorSet();
        // lDirectionalShadowDescriptorSet.Add( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        // auto &lSpotlightShadowDescriptorSet = mPipeline->AddDescriptorSet();
        // lSpotlightShadowDescriptorSet.Add( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        // auto &lPointLightShaadowDescriptorSet = mPipeline->AddDescriptorSet();
        // lPointLightShaadowDescriptorSet.Add( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        mPipeline->Build();

        // SceneRenderPipelineCreateInfo lCreateInfo{};
        // lCreateInfo.VertexShader   = "Shaders/Deferred/DeferredLightingMSAA.vert.spv";
        // lCreateInfo.FragmentShader = "Shaders/Deferred/DeferredLightingMSAA.frag.spv";
        // lCreateInfo.RenderPass     = aCreateInfo.RenderPass;
        // lCreateInfo.DepthTest      = false;
        // lCreateInfo.DepthWrite     = false;

        // Initialize( lCreateInfo );
    }

} // namespace SE::Core