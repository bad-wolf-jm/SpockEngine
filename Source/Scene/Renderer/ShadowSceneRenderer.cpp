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

    void ShadowSceneRenderer::Update( Ref<Scene> aWorld )
    {
        ASceneRenderer::Update( aWorld );

        if( mDirectionalLights.size() != mDirectionalShadowMapRenderContext.size() )
        {
            mDirectionalShadowMapRenderContext.clear();
            mDirectionalShadowMapSamplers.clear();
            for( uint32_t i = 0; i < mDirectionalLights.size(); i++ )
            {
                sRenderTargetDescription lRenderTargetSpec{};
                lRenderTargetSpec.mWidth       = 1024;
                lRenderTargetSpec.mHeight      = 1024;
                lRenderTargetSpec.mSampleCount = 1;
                auto lDirectionalShadowMaps    = New<VkRenderTarget>( mGraphicContext, lRenderTargetSpec );

                sAttachmentDescription lAttachmentCreateInfo{};
                lAttachmentCreateInfo.mIsSampled   = true;
                lAttachmentCreateInfo.mIsPresented = false;
                lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
                lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
                lAttachmentCreateInfo.mType        = eAttachmentType::DEPTH;
                lAttachmentCreateInfo.mClearColor  = { 1.0f, 0.0f, 0.0f, 0.0f };
                lDirectionalShadowMaps->AddAttachment( "SHADOW_MAP", lAttachmentCreateInfo );
                lDirectionalShadowMaps->Finalize();

                mDirectionalShadowMapSamplers.emplace_back();
                mDirectionalShadowMapSamplers.back() = New<Graphics::VkSampler2D>(
                    mGraphicContext, lDirectionalShadowMaps->GetAttachment( "SHADOW_MAP" ), sTextureSamplingInfo{} );

                mDirectionalShadowMapRenderContext.emplace_back( mGraphicContext, lDirectionalShadowMaps );
            }
        }

        if( mSpotlights.size() != mSpotlightShadowMapRenderContext.size() )
        {
            mSpotlightShadowMapRenderContext.clear();
            mSpotlightShadowMapSamplers.clear();
            for( uint32_t i = 0; i < mSpotlights.size(); i++ )
            {
                sRenderTargetDescription lRenderTargetSpec{};
                lRenderTargetSpec.mWidth       = 1024;
                lRenderTargetSpec.mHeight      = 1024;
                lRenderTargetSpec.mSampleCount = 1;
                auto lShadowMaps               = New<VkRenderTarget>( mGraphicContext, lRenderTargetSpec );

                sAttachmentDescription lAttachmentCreateInfo{};
                lAttachmentCreateInfo.mIsSampled   = true;
                lAttachmentCreateInfo.mIsPresented = false;
                lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
                lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
                lAttachmentCreateInfo.mType        = eAttachmentType::DEPTH;
                lAttachmentCreateInfo.mClearColor  = { 1.0f, 0.0f, 0.0f, 0.0f };
                lShadowMaps->AddAttachment( "SHADOW_MAP_1", lAttachmentCreateInfo );
                lShadowMaps->Finalize();

                mSpotlightShadowMapSamplers.emplace_back();
                mSpotlightShadowMapSamplers.back() = New<Graphics::VkSampler2D>(
                    mGraphicContext, lShadowMaps->GetAttachment( "SHADOW_MAP_1" ), sTextureSamplingInfo{} );

                mSpotlightShadowMapRenderContext.emplace_back( mGraphicContext, lShadowMaps );
            }
        }

        if( mDirectionalShadowMapRenderContext.size() > 0 )
        {
            ShadowMeshRendererCreateInfo lCreateInfo{};
            lCreateInfo.RenderPass = mDirectionalShadowMapRenderContext.back().GetRenderPass();
            mRenderPipeline        = ShadowMeshRenderer( mGraphicContext, lCreateInfo );
        }

        if( mSpotlightShadowMapRenderContext.size() > 0 )
        {
            ShadowMeshRendererCreateInfo lCreateInfo{};
            lCreateInfo.RenderPass   = mSpotlightShadowMapRenderContext.back().GetRenderPass();
            mSpotlightRenderPipeline = ShadowMeshRenderer( mGraphicContext, lCreateInfo );
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
                mCameraUniformBuffer->Write( View );

                lContext.BeginRender();
                for( auto &lPipelineData : mOpaqueMeshQueue )
                {
                    if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;

                    lContext.Bind( mRenderPipeline.Pipeline );
                    lContext.Bind( mSceneDescriptors, 0, -1 );
                    lContext.Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );
                    lContext.Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
                }
                lContext.EndRender();
                lLightIndex++;
            }
        }

        if( mSpotlightRenderPipeline.Pipeline )
        {
            uint32_t lLightIndex = 0;
            for( auto &lContext : mSpotlightShadowMapRenderContext )
            {
                View.mMVP = mSpotlights[lLightIndex].Transform;
                mCameraUniformBuffer->Write( View );

                lContext.BeginRender();
                for( auto &lPipelineData : mOpaqueMeshQueue )
                {
                    if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;

                    lContext.Bind( mSpotlightRenderPipeline.Pipeline );
                    lContext.Bind( mSceneDescriptors, 0, -1 );
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