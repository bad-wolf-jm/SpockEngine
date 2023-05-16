#include "ShadowSceneRenderer.h"

#include <chrono>
#include <gli/gli.hpp>

// #include "Graphics/Vulkan/VkPipeline.h"
// #include "Graphics/Vulkan/VkTextureCubeMap.h"

#include "Scene/Components/VisualHelpers.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"

namespace SE::Core
{

    using namespace math;
    using namespace SE::Core::EntityComponentSystem::Components;
    using namespace SE::Core::Primitives;

    Ref<IDescriptorSetLayout> ShadowMeshRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout.AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        lNewLayout.Build();

        return lNewLayout;
    }

    std::vector<Ref<IDescriptorSetLayout>> ShadowMeshRenderer::GetDescriptorSetLayout() { return { CameraSetLayout }; }

    std::vector<sPushConstantRange> ShadowMeshRenderer::GetPushConstantLayout() { return {}; };

    ShadowMeshRenderer::ShadowMeshRenderer( Ref<IGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo )
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

    Ref<IDescriptorSetLayout> OmniShadowMeshRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        DescriptorSetLayoutCreateInfo lCameraBindLayout{};
        lCameraBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } } };

        return New<IDescriptorSetLayout>( aGraphicContext, lCameraBindLayout );
    }

    std::vector<Ref<IDescriptorSetLayout>> OmniShadowMeshRenderer::GetDescriptorSetLayout() { return { CameraSetLayout }; }

    std::vector<sPushConstantRange> OmniShadowMeshRenderer::GetPushConstantLayout() { return {}; };

    OmniShadowMeshRenderer::OmniShadowMeshRenderer( Ref<IGraphicContext>                aGraphicContext,
                                                    ShadowMeshRendererCreateInfo const &aCreateInfo )
        : SceneRenderPipeline<VertexData>( aGraphicContext )
    {

        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.Opaque         = true;
        lCreateInfo.LineWidth      = 1.0f;
        lCreateInfo.VertexShader   = "Shaders\\OmniShadow.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\OmniShadow.frag.spv";
        lCreateInfo.RenderPass     = aCreateInfo.RenderPass;

        CameraSetLayout = GetCameraSetLayout( mGraphicContext );

        Initialize( lCreateInfo );
    }

    ShadowSceneRenderer::ShadowSceneRenderer( Ref<IGraphicContext> aGraphicContext )
        : ASceneRenderer( aGraphicContext, eColorFormat::UNDEFINED, 1 )
    {
        mSceneDescriptors = New<DescriptorSet>( mGraphicContext, ShadowMeshRenderer::GetCameraSetLayout( mGraphicContext ) );

        mCameraUniformBuffer =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
        mSceneDescriptors->Write( mCameraUniformBuffer, false, 0, sizeof( ShadowMatrices ), 0 );
    }

    void ShadowSceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight ) {}

    Ref<IRenderTarget> ShadowSceneRenderer::NewRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = aOutputWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = 1;
        auto lRenderTarget             = CreateRenderTarget( mGraphicContext, lRenderTargetSpec );

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
                mDirectionalShadowMapSamplers.back() =
                    CreateSampler2D( mGraphicContext, lDirectionalShadowMaps->GetAttachment( "SHADOW_MAP" ) );

                mDirectionalShadowMapRenderContext.emplace_back( mGraphicContext, lDirectionalShadowMaps );

                mDirectionalShadowSceneDescriptors.emplace_back();
                mDirectionalShadowSceneDescriptors.back() =
                    New<DescriptorSet>( mGraphicContext, ShadowMeshRenderer::GetCameraSetLayout( mGraphicContext ) );

                mDirectionalShadowCameraUniformBuffer.emplace_back();
                mDirectionalShadowCameraUniformBuffer.back() =
                    CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
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
                mSpotlightShadowMapSamplers.back() = CreateSampler2D( mGraphicContext, lShadowMaps->GetAttachment( "SHADOW_MAP" ) );

                mSpotlightShadowMapRenderContext.emplace_back( mGraphicContext, lShadowMaps );

                mSpotlightShadowSceneDescriptors.emplace_back();
                mSpotlightShadowSceneDescriptors.back() =
                    New<DescriptorSet>( mGraphicContext, ShadowMeshRenderer::GetCameraSetLayout( mGraphicContext ) );

                mSpotlightShadowCameraUniformBuffer.emplace_back();
                mSpotlightShadowCameraUniformBuffer.back() =
                    CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
                mSpotlightShadowSceneDescriptors.back()->Write( mSpotlightShadowCameraUniformBuffer.back(), false, 0,
                                                                sizeof( ShadowMatrices ), 0 );
            }
        }

        if( mPointLights.size() != mPointLightsShadowMapRenderContext.size() )
        {
            mPointLightsShadowMapRenderContext.clear();
            mPointLightsShadowCameraUniformBuffer.clear();
            mPointLightsShadowSceneDescriptors.clear();

            sTextureCreateInfo lCreateInfo{};
            lCreateInfo.mFormat = eColorFormat::R32_FLOAT;
            lCreateInfo.mWidth  = 1024;
            lCreateInfo.mHeight = 1024;
            lCreateInfo.mDepth  = 1;

            for( uint32_t i = 0; i < mPointLights.size(); i++ )
            {
                mPointLightsShadowMapRenderContext.emplace_back();
                mPointLightsShadowSceneDescriptors.emplace_back();
                mPointLightsShadowCameraUniformBuffer.emplace_back();

                auto lShadowMap = New<VkTextureCubeMap>( mGraphicContext, lCreateInfo, 1, false, true, false, false );
                mPointLightShadowMapSamplers.emplace_back();
                mPointLightShadowMapSamplers.back() = New<Graphics::VkSamplerCubeMap>( mGraphicContext, lShadowMap );

                for( uint32_t f = 0; f < 6; f++ )
                {
                    sRenderTargetDescription lRenderTargetSpec{};
                    lRenderTargetSpec.mWidth       = 1024;
                    lRenderTargetSpec.mHeight      = 1024;
                    lRenderTargetSpec.mSampleCount = 1;
                    auto lRenderTarget             = New<IRenderTarget>( mGraphicContext, lRenderTargetSpec );

                    sAttachmentDescription lAttachmentCreateInfo{};
                    lAttachmentCreateInfo.mFormat      = eColorFormat::R32_FLOAT;
                    lAttachmentCreateInfo.mIsSampled   = true;
                    lAttachmentCreateInfo.mIsPresented = false;
                    lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
                    lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
                    lAttachmentCreateInfo.mType        = eAttachmentType::COLOR;
                    lAttachmentCreateInfo.mClearColor  = { 0.0f, 0.0f, 0.0f, 1.0f };
                    lRenderTarget->AddAttachment( "SHADOW_MAP", lAttachmentCreateInfo, lShadowMap, static_cast<eCubeFace>( f ) );

                    lAttachmentCreateInfo              = sAttachmentDescription{};
                    lAttachmentCreateInfo.mIsSampled   = false;
                    lAttachmentCreateInfo.mIsPresented = false;
                    lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
                    lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
                    lAttachmentCreateInfo.mType        = eAttachmentType::DEPTH;
                    lAttachmentCreateInfo.mClearColor  = { 1.0f, 0.0f, 0.0f, 0.0f };
                    lRenderTarget->AddAttachment( "DEPTH", lAttachmentCreateInfo );
                    lRenderTarget->Finalize();

                    mPointLightsShadowMapRenderContext.back()[f] = CreateRenderContext( mGraphicContext, lRenderTarget );
                    mPointLightsShadowSceneDescriptors.back()[f] =
                        New<DescriptorSet>( mGraphicContext, OmniShadowMeshRenderer::GetCameraSetLayout( mGraphicContext ) );

                    mPointLightsShadowCameraUniformBuffer.back()[f] = CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true,
                                                                                    true, true, true, sizeof( OmniShadowMatrices ) );

                    mPointLightsShadowSceneDescriptors.back()[f]->Write( mPointLightsShadowCameraUniformBuffer.back()[f], false, 0,
                                                                         sizeof( OmniShadowMatrices ), 0 );
                }
            }
        }

        if( ( mDirectionalShadowMapRenderContext.size() > 0 ) || ( mSpotlightShadowMapRenderContext.size() > 0 ) )
        {
            ShadowMeshRendererCreateInfo lCreateInfo{};
            lCreateInfo.RenderPass = mDirectionalShadowMapRenderContext.back()->GetRenderPass();
            mRenderPipeline        = ShadowMeshRenderer( mGraphicContext, lCreateInfo );
        }

        if( mPointLightsShadowMapRenderContext.size() > 0 )
        {
            ShadowMeshRendererCreateInfo lCreateInfo{};
            lCreateInfo.RenderPass = mPointLightsShadowMapRenderContext.back()[0]->GetRenderPass();
            mOmniRenderPipeline    = OmniShadowMeshRenderer( mGraphicContext, lCreateInfo );
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

                lContext->BeginRender();
                for( auto &lPipelineData : mOpaqueMeshQueue )
                {
                    if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;

                    lContext->Bind( mRenderPipeline.Pipeline );
                    lContext->Bind( mDirectionalShadowSceneDescriptors[lLightIndex], 0, -1 );
                    lContext->Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );
                    lContext->Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
                }
                lContext->EndRender();
                lLightIndex++;
            }

            lLightIndex = 0;
            for( auto &lContext : mSpotlightShadowMapRenderContext )
            {
                View.mMVP = mSpotlights[lLightIndex].Transform;
                mSpotlightShadowCameraUniformBuffer[lLightIndex]->Write( View );

                lContext->BeginRender();
                for( auto &lPipelineData : mOpaqueMeshQueue )
                {
                    if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;

                    lContext->Bind( mRenderPipeline.Pipeline );
                    lContext->Bind( mSpotlightShadowSceneDescriptors[lLightIndex], 0, -1 );
                    lContext->Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );
                    lContext->Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
                }
                lContext->EndRender();
                lLightIndex++;
            }

            lLightIndex = 0;
            for( auto &lContext : mPointLightsShadowMapRenderContext )
            {

                // clang-format off
                const float aEntries[] = { 1.0f,  0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f };
                math::mat4  lClip = math::MakeMat4( aEntries );
                // clang-format on

                // math::mat4 lProjection = lClip * math::Perspective( math::radians( 90.0f ), 1.0f, .2f, 100.0f );
                math::mat4 lProjection = math::Perspective( math::radians( 90.0f ), 1.0f, .2f, 100.0f );

                // clang-format off
                std::array<math::mat4, 6> lMVPMatrices = {
                    /* POSITIVE_X */ math::LookAt( math::vec3( 0.0f, 0.0f, 0.0f ), math::vec3( 1.0f, 0.0f, 0.0f ), math::vec3( 0.0f, 1.0f, 0.0f )  ),
                    /* NEGATIVE_X */ math::LookAt( math::vec3( 0.0f, 0.0f, 0.0f ), math::vec3( -1.0f, 0.0f, 0.0f ), math::vec3( 0.0f, 1.0f, 0.0f ) ),
                    /* POSITIVE_Y */ math::LookAt( math::vec3( 0.0f, 0.0f, 0.0f ), math::vec3( 0.0f, 1.0f, 0.0f ), math::vec3( 0.0f, 0.0f, -1.0f ) ),
                    /* NEGATIVE_Y */ math::LookAt( math::vec3( 0.0f, 0.0f, 0.0f ), math::vec3( 0.0f, -1.0f, 0.0f ), math::vec3( 0.0f, 0.0f, 1.0f ) ),
                    /* POSITIVE_Z */ math::LookAt( math::vec3( 0.0f, 0.0f, 0.0f ), math::vec3( 0.0f, 0.0f, 1.0f ), math::vec3( 0.0f, 1.0f, 0.0f ) ),
                    /* NEGATIVE_Z */ math::LookAt( math::vec3( 0.0f, 0.0f, 0.0f ), math::vec3( 0.0f, 0.0f, -1.0f ), math::vec3( 0.0f, 1.0f, 0.0f ) ),
                };
                // clang-format on

                for( uint32_t f = 0; f < 6; f++ )
                {

                    glm::mat4 viewMatrix = glm::mat4( 1.0f );
                    switch( f )
                    {
                    case 0: // POSITIVE_X
                        viewMatrix = glm::rotate( viewMatrix, glm::radians( 90.0f ), glm::vec3( 0.0f, 1.0f, 0.0f ) );
                        viewMatrix = glm::rotate( viewMatrix, glm::radians( 180.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
                        break;
                    case 1: // NEGATIVE_X
                        viewMatrix = glm::rotate( viewMatrix, glm::radians( -90.0f ), glm::vec3( 0.0f, 1.0f, 0.0f ) );
                        viewMatrix = glm::rotate( viewMatrix, glm::radians( 180.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
                        break;
                    case 2: // POSITIVE_Y
                        viewMatrix = glm::rotate( viewMatrix, glm::radians( -90.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
                        break;
                    case 3: // NEGATIVE_Y
                        viewMatrix = glm::rotate( viewMatrix, glm::radians( 90.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
                        break;
                    case 4: // POSITIVE_Z
                        viewMatrix = glm::rotate( viewMatrix, glm::radians( 180.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
                        break;
                    case 5: // NEGATIVE_Z
                        viewMatrix = glm::rotate( viewMatrix, glm::radians( 180.0f ), glm::vec3( 0.0f, 0.0f, 1.0f ) );
                        break;
                    }

                    mOmniView.mMVP = lProjection * math::Translate( viewMatrix, -mPointLights[lLightIndex].WorldPosition );
                    ;
                    mOmniView.mLightPos = math::vec4( mPointLights[lLightIndex].WorldPosition, 0.0f );
                    mPointLightsShadowCameraUniformBuffer[lLightIndex][f]->Write( mOmniView );

                    lContext[f]->BeginRender();
                    for( auto &lPipelineData : mOpaqueMeshQueue )
                    {
                        if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;

                        lContext[f]->Bind( mOmniRenderPipeline.Pipeline );
                        lContext[f]->Bind( mPointLightsShadowSceneDescriptors[lLightIndex][f], 0, -1 );
                        lContext[f]->Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );
                        lContext[f]->Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
                    }
                    lContext[f]->EndRender();
                }

                lLightIndex++;
            }
        }
    }

    Ref<ITexture2D> ShadowSceneRenderer::GetOutputImage()
    {
        //
        return mGeometryRenderTarget->GetAttachment( "OUTPUT" );
    }
} // namespace SE::Core