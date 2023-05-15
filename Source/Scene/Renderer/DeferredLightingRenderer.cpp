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
    // Ref<DescriptorSetLayout> DeferredLightingRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    // {
    //     DescriptorSetLayoutCreateInfo l_CameraBindLayout{};
    //     l_CameraBindLayout.Bindings = {
    //         DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } },
    //         DescriptorBindingInfo{ 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } } };

    //     return New<DescriptorSetLayout>( aGraphicContext, l_CameraBindLayout );
    // }

    // Ref<DescriptorSetLayout> DeferredLightingRenderer::GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext )
    // {
    //     DescriptorSetLayoutCreateInfo lTextureBindLayout{};
    //     lTextureBindLayout.Bindings = {
    //         DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } },
    //         DescriptorBindingInfo{ 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } },
    //         DescriptorBindingInfo{ 2, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } },
    //         DescriptorBindingInfo{ 3, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

    //     return New<DescriptorSetLayout>( aGraphicContext, lTextureBindLayout, false );
    // }

    // Ref<DescriptorSetLayout> DeferredLightingRenderer::GetDirectionalShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    // {
    //     DescriptorSetLayoutCreateInfo lShadowMapLayout{};
    //     lShadowMapLayout.Bindings = {
    //         DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

    //     return New<DescriptorSetLayout>( aGraphicContext, lShadowMapLayout, true );
    // }

    // Ref<DescriptorSetLayout> DeferredLightingRenderer::GetSpotlightShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    // {
    //     DescriptorSetLayoutCreateInfo lShadowMapLayout{};
    //     lShadowMapLayout.Bindings = {
    //         DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

    //     return New<DescriptorSetLayout>( aGraphicContext, lShadowMapLayout, true );
    // }

    // Ref<DescriptorSetLayout> DeferredLightingRenderer::GetPointLightShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    // {
    //     DescriptorSetLayoutCreateInfo lShadowMapLayout{};
    //     lShadowMapLayout.Bindings = {
    //         DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

    //     return New<DescriptorSetLayout>( aGraphicContext, lShadowMapLayout, true );
    // }

    // std::vector<Ref<DescriptorSetLayout>> DeferredLightingRenderer::GetDescriptorSetLayout()
    // {
    //     return { CameraSetLayout, TextureSetLayout, DirectionalShadowSetLayout, SpotlightShadowSetLayout, PointLightShadowSetLayout };
    // }

    // std::vector<sPushConstantRange> DeferredLightingRenderer::GetPushConstantLayout()
    // {
    //     return { { { eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) } };
    // };

    DeferredLightingRenderer::DeferredLightingRenderer( Ref<IGraphicContext> mGraphicContext, Ref<IRenderContext> aRenderContext )
        : mGraphicContext( aGraphicContext )
    {

        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::NONE );
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, GetResourcePath( "Shaders/Deferred/DeferredLightingMSAA.vert.spv" ),
                              "main" );
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, GetResourcePath( "Shaders/Deferred/DeferredLightingMSAA.frag.spv" ),
                              "main" );
        mPipeline->AddPushConstantRange( { eShaderStageTypeFlags::FRAGMENT }, 0, sizeof( MaterialPushConstants ) );

        auto &lCameraDescriptorSet = mPipeline->AddDescriptorSet();
        lCameraDescriptorSet.Add( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lCameraDescriptorSet.Add( 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );

        auto &lTextureDescriptorSet = mPipeline->AddDescriptorSet();
        lTextureDescriptorSet.Add( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lTextureDescriptorSet.Add( 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lTextureDescriptorSet.Add( 2, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lTextureDescriptorSet.Add( 3, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        auto &lDirectionalShadowDescriptorSet = mPipeline->AddDescriptorSet();
        lDirectionalShadowDescriptorSet.Add( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        auto &lSpotlightShadowDescriptorSet = mPipeline->AddDescriptorSet();
        lSpotlightShadowDescriptorSet.Add( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        auto &lPointLightShaadowDescriptorSet = mPipeline->AddDescriptorSet();
        lPointLightShaadowDescriptorSet.Add( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        mPipeline->Build();

        // SceneRenderPipelineCreateInfo lCreateInfo{};
        // lCreateInfo.VertexShader   = "Shaders/Deferred/DeferredLightingMSAA.vert.spv";
        // lCreateInfo.FragmentShader = "Shaders/Deferred/DeferredLightingMSAA.frag.spv";
        // lCreateInfo.RenderPass     = aCreateInfo.RenderPass;
        // lCreateInfo.DepthTest      = false;
        // lCreateInfo.DepthWrite     = false;

        // CameraSetLayout            = GetCameraSetLayout( mGraphicContext );
        // TextureSetLayout           = GetTextureSetLayout( mGraphicContext );
        // DirectionalShadowSetLayout = GetDirectionalShadowSetLayout( mGraphicContext );
        // SpotlightShadowSetLayout   = GetSpotlightShadowSetLayout( mGraphicContext );
        // PointLightShadowSetLayout  = GetPointLightShadowSetLayout( mGraphicContext );

        // Initialize( lCreateInfo );
    }

} // namespace SE::Core