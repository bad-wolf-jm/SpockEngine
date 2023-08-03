#include "ShadowSceneRenderer.h"

#include <chrono>
#include <gli/gli.hpp>

#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

#include "Shaders/gDirectionalShadowVertexShader.h"
#include "Shaders/gOmniDirectionalShadowFragmentShader.h"
#include "Shaders/gOmniDirectionalShadowVertexShader.h"
#include "Shaders/gVertexLayout.h"
namespace SE::Core
{
    using namespace math;
    using namespace SE::Core::EntityComponentSystem::Components;
    using namespace SE::Core::Primitives;

    Ref<IDescriptorSetLayout> ShadowMeshRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        lNewLayout->Build();

        return lNewLayout;
    }

    ShadowMeshRenderer::ShadowMeshRenderer( Ref<IGraphicContext> aGraphicContext, ShadowMeshRendererCreateInfo const &aCreateInfo )
        : mGraphicContext( aGraphicContext )
        , Spec{ aCreateInfo }
    {

        mPipeline = CreateGraphicsPipeline( mGraphicContext, Spec.RenderPass, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::BACK );
        mPipeline->SetDepthParameters( true, true, eDepthCompareOperation::LESS_OR_EQUAL );

        fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
        auto     lVertexShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, "shadow_vertex_shader", lShaderPath );
        lVertexShader->AddCode( SE::Private::Shaders::gVertexLayout_data );
        lVertexShader->AddCode( SE::Private::Shaders::gDirectionalShadowVertexShader_data );
        lVertexShader->Compile();

        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );
        mPipeline->AddInput( "Position", eBufferDataType::VEC3, 0, 0 );
        mPipeline->AddInput( "Normal", eBufferDataType::VEC3, 0, 1 );
        mPipeline->AddInput( "TexCoord_0", eBufferDataType::VEC2, 0, 2 );
        mPipeline->AddInput( "Bones", eBufferDataType::VEC4, 0, 3 );
        mPipeline->AddInput( "Weights", eBufferDataType::VEC4, 0, 4 );

        CameraSetLayout = GetCameraSetLayout( mGraphicContext );

        mPipeline->AddDescriptorSet( CameraSetLayout );

        mPipeline->Build();
    }

    Ref<IDescriptorSetLayout> OmniShadowMeshRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } );
        lNewLayout->Build();

        return lNewLayout;
    }

    OmniShadowMeshRenderer::OmniShadowMeshRenderer( Ref<IGraphicContext>                aGraphicContext,
                                                    ShadowMeshRendererCreateInfo const &aCreateInfo )
        : mGraphicContext( aGraphicContext )
        , Spec{ aCreateInfo }
    {
        mPipeline = CreateGraphicsPipeline( mGraphicContext, Spec.RenderPass, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::BACK );
        mPipeline->SetDepthParameters( true, true, eDepthCompareOperation::LESS_OR_EQUAL );

        fs::path lShaderPath = "D:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
        auto     lVertexShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, "omni_shadow_vertex_shader", lShaderPath );
        lVertexShader->AddCode( SE::Private::Shaders::gVertexLayout_data );
        lVertexShader->AddCode( SE::Private::Shaders::gOmniDirectionalShadowVertexShader_data );
        lVertexShader->Compile();
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );

        auto lFragmentShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::FRAGMENT, 450, "omni_shadow_fragment_shader", lShaderPath );
        lFragmentShader->AddCode( SE::Private::Shaders::gOmniDirectionalShadowFragmentShader_data );
        lFragmentShader->Compile();

        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, lFragmentShader, "main" );
        mPipeline->AddInput( "Position", eBufferDataType::VEC3, 0, 0 );
        mPipeline->AddInput( "Normal", eBufferDataType::VEC3, 0, 1 );
        mPipeline->AddInput( "TexCoord_0", eBufferDataType::VEC2, 0, 2 );
        mPipeline->AddInput( "Bones", eBufferDataType::VEC4, 0, 3 );
        mPipeline->AddInput( "Weights", eBufferDataType::VEC4, 0, 4 );

        CameraSetLayout = GetCameraSetLayout( mGraphicContext );

        mPipeline->AddDescriptorSet( CameraSetLayout );
        mPipeline->Build();
    }

    static math::mat4 CreateCubeFaceViewMatrix( int aFace )
    {
        glm::mat4 lViewMatrix = glm::mat4( 1.0f );
        switch( aFace )
        {
        case 0:
            // POSITIVE_X
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 90.0f ), glm::vec3( 0.0f, 1.0f, 0.0f ) );
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 180.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 1:
            // NEGATIVE_X
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( -90.0f ), glm::vec3( 0.0f, 1.0f, 0.0f ) );
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 180.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 2:
            // POSITIVE_Y
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( -90.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 3:
            // NEGATIVE_Y
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 90.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 4:
            // POSITIVE_Z
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 180.0f ), glm::vec3( 1.0f, 0.0f, 0.0f ) );
            break;
        case 5:
            // NEGATIVE_Z
            lViewMatrix = glm::rotate( lViewMatrix, glm::radians( 180.0f ), glm::vec3( 0.0f, 0.0f, 1.0f ) );
            break;
        }

        return lViewMatrix;
    }

    ShadowSceneRenderer::ShadowSceneRenderer( Ref<IGraphicContext> aGraphicContext )
        : BaseSceneRenderer( aGraphicContext, eColorFormat::UNDEFINED, 1 )
    {
        auto lLayout      = ShadowMeshRenderer::GetCameraSetLayout( mGraphicContext );
        mSceneDescriptors = lLayout->Allocate();

        mCameraSetLayout = ShadowMeshRenderer::GetCameraSetLayout( mGraphicContext );

        mCameraUniformBuffer =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
        mSceneDescriptors->Write( mCameraUniformBuffer, false, 0, sizeof( ShadowMatrices ), 0 );

        auto lDirectionalShadowMaps        = NewRenderTarget( 1024, 1024 );
        mDirectionalShadowMapRenderContext = CreateRenderContext( mGraphicContext, lDirectionalShadowMaps );
        ShadowMeshRendererCreateInfo lCreateInfo{};
        lCreateInfo.RenderPass = mDirectionalShadowMapRenderContext;

        mRenderPipeline                   = New<ShadowMeshRenderer>( mGraphicContext, lCreateInfo );
        mDirectionalShadowMapSampler      = CreateSampler2D( mGraphicContext, lDirectionalShadowMaps->GetAttachment( "SHADOW_MAP" ) );
        mDirectionalShadowSceneDescriptor = mCameraSetLayout->Allocate();
        mDirectionalShadowCameraUniformBuffer =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( ShadowMatrices ) );
        mDirectionalShadowSceneDescriptor->Write( mDirectionalShadowCameraUniformBuffer, false, 0, sizeof( ShadowMatrices ), 0 );
    }

    void ShadowSceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
    }

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

    void ShadowSceneRenderer::SetLights( sDirectionalLight const &aDirectionalLights )
    {
        mDirectionalLight = aDirectionalLights;
    }

    void ShadowSceneRenderer::SetLights( std::vector<sPunctualLight> const &aPointLights )
    {
        mPointLights = aPointLights;
    }

    void ShadowSceneRenderer::Update( Ref<Scene> aWorld )
    {
        BaseSceneRenderer::Update( aWorld );

        constexpr int32_t mOmniShadowResolution = 1024;
        if( mPointLights.size() != mPointLightsShadowMapRenderContext.size() )
        {
            mPointLightsShadowMapRenderContext.clear();
            mPointLightsShadowCameraUniformBuffer.clear();
            mPointLightsShadowSceneDescriptors.clear();

            sTextureCreateInfo lCreateInfo{};
            lCreateInfo.mFormat = eColorFormat::RGBA32_FLOAT;
            lCreateInfo.mWidth  = mOmniShadowResolution;
            lCreateInfo.mHeight = mOmniShadowResolution;
            lCreateInfo.mDepth  = 1;
            lCreateInfo.mLayers = 6;

            for( uint32_t i = 0; i < mPointLights.size(); i++ )
            {
                mPointLightsShadowMapRenderContext.emplace_back();
                mPointLightsShadowSceneDescriptors.emplace_back();
                mPointLightsShadowCameraUniformBuffer.emplace_back();

                auto lShadowMap = CreateTexture2D( mGraphicContext, lCreateInfo, 1, false, true, false, false );
                mPointLightShadowMapSamplers.emplace_back();
                mPointLightShadowMapSamplers.back() = CreateSamplerCubeMap( mGraphicContext, lShadowMap );

                for( uint32_t f = 0; f < 6; f++ )
                {
                    sRenderTargetDescription lRenderTargetSpec{};
                    lRenderTargetSpec.mWidth       = mOmniShadowResolution;
                    lRenderTargetSpec.mHeight      = mOmniShadowResolution;
                    lRenderTargetSpec.mSampleCount = 1;
                    auto lRenderTarget             = CreateRenderTarget( mGraphicContext, lRenderTargetSpec );

                    sAttachmentDescription lAttachmentCreateInfo{};
                    lAttachmentCreateInfo.mFormat      = eColorFormat::R32_FLOAT;
                    lAttachmentCreateInfo.mFormat      = eColorFormat::RGBA32_FLOAT;
                    lAttachmentCreateInfo.mIsSampled   = false;
                    lAttachmentCreateInfo.mIsPresented = false;
                    lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
                    lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
                    lAttachmentCreateInfo.mType        = eAttachmentType::COLOR;
                    lAttachmentCreateInfo.mClearColor  = { .0f, .0f, .0f, 1.f };
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

                    mPointLightsShadowMapRenderContext.back()[f]    = CreateRenderContext( mGraphicContext, lRenderTarget );
                    mPointLightsShadowSceneDescriptors.back()[f]    = mCameraSetLayout->Allocate();
                    mPointLightsShadowCameraUniformBuffer.back()[f] = CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true,
                                                                                    true, true, true, sizeof( OmniShadowMatrices ) );
                    mPointLightsShadowSceneDescriptors.back()[f]->Write( mPointLightsShadowCameraUniformBuffer.back()[f], false, 0,
                                                                         sizeof( OmniShadowMatrices ), 0 );
                }
            }
        }

        if( mOmniRenderPipeline == nullptr && mPointLightsShadowMapRenderContext.size() > 0 )
        {
            ShadowMeshRendererCreateInfo lCreateInfo{};
            lCreateInfo.RenderPass = mPointLightsShadowMapRenderContext.back()[0];
            mOmniRenderPipeline    = New<OmniShadowMeshRenderer>( mGraphicContext, lCreateInfo );
        }
    }

    void ShadowSceneRenderer::Render()
    {
        SE_PROFILE_FUNCTION();

        if( !mScene )
            return;

        if( mRenderPipeline->Pipeline() )
        {
            uint32_t lLightIndex = 0;
            View.mMVP            = mDirectionalLight.mTransform;
            mDirectionalShadowCameraUniformBuffer->Write( View );

            mDirectionalShadowMapRenderContext->BeginRender();
            mDirectionalShadowMapRenderContext->Bind( mRenderPipeline->Pipeline() );
            mDirectionalShadowMapRenderContext->Bind( mDirectionalShadowSceneDescriptor, 0, -1 );

            mScene->ForEach<sStaticMeshComponent, sMaterialComponent>(
                [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterial )
                {
                    if( !aStaticMeshComponent.mVertexBuffer || !aStaticMeshComponent.mIndexBuffer )
                        return;

                    mDirectionalShadowMapRenderContext->Bind( aStaticMeshComponent.mTransformedBuffer,
                                                              aStaticMeshComponent.mIndexBuffer );
                    mDirectionalShadowMapRenderContext->Draw( aStaticMeshComponent.mIndexCount, aStaticMeshComponent.mIndexOffset,
                                                              aStaticMeshComponent.mVertexOffset, 1, 0 );
                } );

            mDirectionalShadowMapRenderContext->EndRender();

            lLightIndex = 0;
            for( auto &lContext : mPointLightsShadowMapRenderContext )
            {
                math::mat4 lProjection = math::Perspective( math::radians( 90.0f ), 1.0f, .2f, 1000.0f );
                mOmniView.mLightPos    = math::vec4( mPointLights[lLightIndex].mPosition, 0.0f );

                for( uint32_t f = 0; f < 6; f++ )
                {
                    glm::mat4 viewMatrix = CreateCubeFaceViewMatrix( f );
                    mOmniView.mMVP       = lProjection * math::Translate( viewMatrix, -mPointLights[lLightIndex].mPosition );
                    mPointLightsShadowCameraUniformBuffer[lLightIndex][f]->Write( mOmniView );

                    lContext[f]->BeginRender();
                    lContext[f]->Bind( mOmniRenderPipeline->Pipeline() );
                    lContext[f]->Bind( mPointLightsShadowSceneDescriptors[lLightIndex][f], 0, -1 );

                    mScene->ForEach<sStaticMeshComponent, sMaterialComponent>(
                        [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterial )
                        {
                            if( !aStaticMeshComponent.mVertexBuffer || !aStaticMeshComponent.mIndexBuffer )
                                return;

                            lContext[f]->Bind( aStaticMeshComponent.mTransformedBuffer, aStaticMeshComponent.mIndexBuffer );
                            lContext[f]->Draw( aStaticMeshComponent.mIndexCount, aStaticMeshComponent.mIndexOffset,
                                               aStaticMeshComponent.mVertexOffset, 1, 0 );
                        } );

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