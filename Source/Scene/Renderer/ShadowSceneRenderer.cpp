#include "ShadowSceneRenderer.h"

#include <chrono>
#include <gli/gli.hpp>

#include "Graphics/Vulkan/VkPipeline.h"

#include "Scene/Components/VisualHelpers.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Profiling/BlockTimer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"

namespace SE::Core
{

    using namespace math;
    using namespace SE::Core::EntityComponentSystem::Components;
    using namespace SE::Core::Primitives;

    Ref<DescriptorSetLayout> ShadowMeshRenderer::GetCameraSetLayout( Ref<VkGraphicContext> aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo lCameraBindLayout{};
        lCameraBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } } };

        return New<DescriptorSetLayout>( aGraphicContext, lCameraBindLayout );
    }

    std::vector<Ref<DescriptorSetLayout>> ShadowMeshRenderer::GetDescriptorSetLayout() { return { CameraSetLayout }; }

    std::vector<sPushConstantRange> ShadowMeshRenderer::GetPushConstantLayout() { return {}; };

    ShadowMeshRenderer::ShadowMeshRenderer( Ref<VkGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo )
        : SceneRenderPipeline<VertexData>( aGraphicContext )
    {

        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.Opaque       = true;
        lCreateInfo.LineWidth    = 1.0f;
        lCreateInfo.VertexShader = "Shaders\\Shadow.vert.spv";
        lCreateInfo.RenderPass   = aCreateInfo.RenderPass;

        CameraSetLayout = GetCameraSetLayout( mGraphicContext );

        Initialize( lCreateInfo );
    }

    ShadowSceneRenderer::ShadowSceneRenderer( Ref<VkGraphicContext> aGraphicContext )
        : ASceneRenderer( aGraphicContext, eColorFormat::UNDEFINED, 1 )
    {
        mSceneDescriptors = New<DescriptorSet>( mGraphicContext, ShadowMeshRenderer::GetCameraSetLayout( mGraphicContext ) );

        mCameraUniformBuffer =
            New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
        mSceneDescriptors->Write( mCameraUniformBuffer, false, 0, sizeof( ShadowMatrices ), 0 );
    }

    void ShadowSceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight ) {}

    Ref<VkRenderTarget> ShadowSceneRenderer::NewRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = aOutputWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = 1;
        auto lRenderTarget             = New<VkRenderTarget>( mGraphicContext, lRenderTargetSpec );

        sAttachmentDescription lAttachmentCreateInfo{};
        lAttachmentCreateInfo.mIsSampled   = true;
        lAttachmentCreateInfo.mIsPresented = false;
        lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
        lAttachmentCreateInfo.mType        = eAttachmentType::DEPTH;
        lAttachmentCreateInfo.mClearColor  = { 1.0f, 0.0f, 0.0f, 0.0f };
        lRenderTarget->AddAttachment( "SHADOW_MAP", lAttachmentCreateInfo );
        lRenderTarget->Finalize();

        return lRenderTarget;
    }

    void ShadowSceneRenderer::Update( Ref<Scene> aWorld )
    {
        ASceneRenderer::Update( aWorld );

        if( mDirectionalLights.size() != mDirectionalShadowMapRenderContext.size() )
        {
            mDirectionalShadowMapRenderContext.clear();
            mDirectionalShadowMapSamplers.clear();
            for( uint32_t i = 0; i < mDirectionalLights.size(); i++ )
            {
                auto lDirectionalShadowMaps = NewRenderTarget( 1024, 1024 );

                mDirectionalShadowMapSamplers.emplace_back();
                mDirectionalShadowMapSamplers.back() = New<Graphics::VkSampler2D>(
                    mGraphicContext, lDirectionalShadowMaps->GetAttachment( "SHADOW_MAP" ), sTextureSamplingInfo{} );

                mDirectionalShadowMapRenderContext.emplace_back( mGraphicContext, lDirectionalShadowMaps );

                mDirectionalShadowSceneDescriptors.emplace_back();
                mDirectionalShadowSceneDescriptors.back() =
                    New<DescriptorSet>( mGraphicContext, ShadowMeshRenderer::GetCameraSetLayout( mGraphicContext ) );

                mDirectionalShadowCameraUniformBuffer.emplace_back();
                mDirectionalShadowCameraUniformBuffer.back() =
                    New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
                mDirectionalShadowSceneDescriptors.back()->Write( mDirectionalShadowCameraUniformBuffer.back(), false, 0,
                                                                  sizeof( ShadowMatrices ), 0 );
            }
        }

        if( mSpotlights.size() != mSpotlightShadowMapRenderContext.size() )
        {
            mSpotlightShadowMapRenderContext.clear();
            mSpotlightShadowMapSamplers.clear();
            for( uint32_t i = 0; i < mSpotlights.size(); i++ )
            {
                auto lShadowMaps = NewRenderTarget( 1024, 1024 );

                mSpotlightShadowMapSamplers.emplace_back();
                mSpotlightShadowMapSamplers.back() =
                    New<Graphics::VkSampler2D>( mGraphicContext, lShadowMaps->GetAttachment( "SHADOW_MAP" ), sTextureSamplingInfo{} );

                mSpotlightShadowMapRenderContext.emplace_back( mGraphicContext, lShadowMaps );

                mSpotlightShadowSceneDescriptors.emplace_back();
                mSpotlightShadowSceneDescriptors.back() =
                    New<DescriptorSet>( mGraphicContext, ShadowMeshRenderer::GetCameraSetLayout( mGraphicContext ) );

                mSpotlightShadowCameraUniformBuffer.emplace_back();
                mSpotlightShadowCameraUniformBuffer.back() =
                    New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
                mSpotlightShadowSceneDescriptors.back()->Write( mSpotlightShadowCameraUniformBuffer.back(), false, 0,
                                                                sizeof( ShadowMatrices ), 0 );
            }
        }

        if( mPointLights.size() != mPointLightsShadowMapRenderContext.size() )
        {
            mPointLightsShadowMapRenderContext.clear();
            for( uint32_t i = 0; i < mPointLights.size(); i++ )
            {
                for( uint32_t f = 0; f < 6; f++ )
                {
                    auto lShadowMaps = NewRenderTarget( 1024, 1024 );

                    mPointLightsShadowMapRenderContext.emplace_back( mGraphicContext, lShadowMaps );

                    mPointLightsShadowSceneDescriptors.emplace_back();
                    mPointLightsShadowSceneDescriptors.back() =
                        New<DescriptorSet>( mGraphicContext, ShadowMeshRenderer::GetCameraSetLayout( mGraphicContext ) );

                    mPointLightsShadowCameraUniformBuffer.emplace_back();
                    mPointLightsShadowCameraUniformBuffer.back() = New<VkGpuBuffer>(
                        mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
                    mPointLightsShadowSceneDescriptors.back()->Write( mPointLightsShadowCameraUniformBuffer.back(), false, 0,
                                                                      sizeof( ShadowMatrices ), 0 );
                }
            }
        }

        if( ( mDirectionalShadowMapRenderContext.size() > 0 ) || ( mSpotlightShadowMapRenderContext.size() > 0 ) )
        {
            ShadowMeshRendererCreateInfo lCreateInfo{};
            lCreateInfo.RenderPass = mDirectionalShadowMapRenderContext.back().GetRenderPass();
            mRenderPipeline        = ShadowMeshRenderer( mGraphicContext, lCreateInfo );
        }
    }

    void ShadowSceneRenderer::Render()
    {
        SE_PROFILE_FUNCTION();

        if( !mScene ) return;

        if( mRenderPipeline.Pipeline )
        {
            uint32_t lLightIndex = 0;
            for( auto &lContext : mDirectionalShadowMapRenderContext )
            {
                View.mMVP = mDirectionalLights[lLightIndex].Transform;
                mDirectionalShadowCameraUniformBuffer[lLightIndex]->Write( View );

                lContext.BeginRender();
                for( auto &lPipelineData : mOpaqueMeshQueue )
                {
                    if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;

                    lContext.Bind( mRenderPipeline.Pipeline );
                    lContext.Bind( mDirectionalShadowSceneDescriptors[lLightIndex], 0, -1 );
                    lContext.Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );
                    lContext.Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
                }
                lContext.EndRender();
                lLightIndex++;
            }

            lLightIndex = 0;
            for( auto &lContext : mSpotlightShadowMapRenderContext )
            {
                View.mMVP = mSpotlights[lLightIndex].Transform;
                mSpotlightShadowCameraUniformBuffer[lLightIndex]->Write( View );

                lContext.BeginRender();
                for( auto &lPipelineData : mOpaqueMeshQueue )
                {
                    if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;

                    lContext.Bind( mRenderPipeline.Pipeline );
                    lContext.Bind( mSpotlightShadowSceneDescriptors[lLightIndex], 0, -1 );
                    lContext.Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );
                    lContext.Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
                }
                lContext.EndRender();
                lLightIndex++;
            }
        }
    }

    Ref<VkTexture2D> ShadowSceneRenderer::GetOutputImage()
    {
        //
        return mGeometryRenderTarget->GetAttachment( "OUTPUT" );
    }
} // namespace SE::Core