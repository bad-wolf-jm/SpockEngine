#include "NewSceneRenderer.h"

#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

#include "Renderer2/Common/LightInputData.hpp"
#include "Scene/Scene.h"

namespace SE::Core
{
    using namespace SE::Core::EntityComponentSystem::Components;
    using namespace SE::Core::Primitives;
    using namespace math;

    NewSceneRenderer::NewSceneRenderer( Ref<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
        : BaseSceneRenderer( aGraphicContext, aOutputFormat, aOutputSampleCount )
    {
    }

    void NewSceneRenderer::CreateRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = aOutputWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = mOutputSampleCount;
        mGeometryRenderTarget          = Graphics::CreateRenderTarget( mGraphicContext, lRenderTargetSpec );

        sAttachmentDescription lAttachmentCreateInfo{};
        lAttachmentCreateInfo.mIsSampled   = true;
        lAttachmentCreateInfo.mIsPresented = false;
        lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;

        lAttachmentCreateInfo.mType       = eAttachmentType::COLOR;
        lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA8_UNORM;
        lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };

        mGeometryRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
        lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
        mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );
        mGeometryRenderTarget->Finalize();
        mGeometryContext = CreateRenderContext( mGraphicContext, mGeometryRenderTarget );
    }

    void NewSceneRenderer::CreateMSAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = aOutputWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = mOutputSampleCount;
        mGeometryRenderTarget          = Graphics::CreateRenderTarget( mGraphicContext, lRenderTargetSpec );

        sAttachmentDescription lAttachmentCreateInfo{};
        lAttachmentCreateInfo.mIsSampled   = true;
        lAttachmentCreateInfo.mIsPresented = false;
        lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;

        lAttachmentCreateInfo.mType       = eAttachmentType::COLOR;
        lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA8_UNORM;
        lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
        mGeometryRenderTarget->AddAttachment( "MSAA_OUTPUT", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mType       = eAttachmentType::MSAA_RESOLVE;
        lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA8_UNORM;
        lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
        mGeometryRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
        lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
        mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );

        mGeometryRenderTarget->Finalize();
        mGeometryContext = CreateRenderContext( mGraphicContext, mGeometryRenderTarget );
    }

    void NewSceneRenderer::CreateFXAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lFxaaSpec{};
        lFxaaSpec.mWidth       = aOutputWidth;
        lFxaaSpec.mHeight      = aOutputHeight;
        lFxaaSpec.mSampleCount = mOutputSampleCount;
        mFxaaRenderTarget      = Graphics::CreateRenderTarget( mGraphicContext, lFxaaSpec );

        sAttachmentDescription lAttachmentCreateInfo{};
        lAttachmentCreateInfo.mIsSampled   = true;
        lAttachmentCreateInfo.mIsPresented = false;
        lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
        lAttachmentCreateInfo.mType        = eAttachmentType::COLOR;
        lAttachmentCreateInfo.mFormat      = eColorFormat::RGBA16_FLOAT;
        lAttachmentCreateInfo.mClearColor  = { 0.0f, 0.0f, 0.0f, 1.0f };
        lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;

        mFxaaRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );
        mFxaaRenderTarget->Finalize();

        mFxaaContext = CreateRenderContext( mGraphicContext, mFxaaRenderTarget );
    }

    void NewSceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        if( mOutputSampleCount == 1 )
            CreateRenderTarget( aOutputWidth, aOutputHeight );
        else
            CreateMSAARenderTarget( aOutputWidth, aOutputHeight );
        mShadowSceneRenderer = New<NewShadowSceneRenderer>( mGraphicContext );
    }

    void NewSceneRenderer::Update( Ref<Scene> aWorld )
    {
        SE_PROFILE_FUNCTION();

        BaseSceneRenderer::Update( aWorld );

        if( mScene == nullptr )
            return;

        bool                        lFoundDirectionalLight = false;
        sDirectionalLight           lDirectionalLight{};
        std::vector<sPunctualLight> lPointLights;

        // clang-format off
        const float aEntries[] = { 1.0f,  0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f };
        mScene->ForEach<sLightComponent>( [&]( auto aEntity, auto &aComponent ) 
        { 
            switch( aComponent.mType )
            {
            case eLightType::DIRECTIONAL:
                if (lFoundDirectionalLight)
                    return;
                if ((!aComponent.mIsOn))
                    return;
                lDirectionalLight.mColorIntensity = math::vec4(aComponent.mColor, aComponent.mIntensity);
                lDirectionalLight.mDirection = mat3(mScene->GetFinalTransformMatrix( aEntity ) )* vec3{ 0.0f, 0.0f, 1.0f };
                lDirectionalLight.mCastsShadows = 1;
                
                math::mat4  lClip = math::MakeMat4( aEntries );
                math::mat4 lProjection =
                    math::Orthogonal( math::vec2{ -10.0f, 10.0f }, math::vec2{ -10.0f, 10.0f }, math::vec2{ -10.0f, 10.0f } );
                math::mat4 lView =
                    math::LookAt( lDirectionalLight.mDirection * 5.0f, math::vec3{ 0.0f, 0.0f, 0.0f }, math::vec3{ 0.0f, 1.0f, 0.0f } );
                lDirectionalLight.mTransform = lClip * lProjection * lView;

                lFoundDirectionalLight = true;
                break;
            case eLightType::POINT_LIGHT:
                if( ( !aComponent.mIsOn ) )
                    return;
                auto &lNewPointLight           = lPointLights.emplace_back();
                lNewPointLight.mColorIntensity = math::vec4( aComponent.mColor, aComponent.mIntensity );
                lNewPointLight.mPosition       = vec3( mScene->GetFinalTransformMatrix( aEntity )[3] );
                lNewPointLight.mCastsShadows   = 1;
                break;
                }
            } );
        // clang-format on

        mScene->GetNewMaterialSystem()->SetLights( lDirectionalLight );
        mScene->GetNewMaterialSystem()->SetLights( lPointLights );

        mShadowSceneRenderer->SetLights( lDirectionalLight );
        mShadowSceneRenderer->SetLights( lPointLights );

        for( auto &[lMaterialHash, lRenderQueue] : mPipelines )
            lRenderQueue.mMeshes.clear();

        mScene->ForEach<sStaticMeshComponent, sNewMaterialComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterial )
            {
                size_t lMaterialHash = mScene->GetNewMaterialSystem()->GetMaterialHash( aMaterial.mMaterialID );
                if( mPipelines.find( lMaterialHash ) == mPipelines.end() )
                {
                    mPipelines.emplace( lMaterialHash, sRenderQueue{} );
                    mPipelines[lMaterialHash].mPipeline =
                        mScene->GetNewMaterialSystem()->CreateGraphicsPipeline( aMaterial.mMaterialID, mGeometryContext );
                }
                mPipelines[lMaterialHash].mMeshes.emplace_back( aStaticMeshComponent, aMaterial );
            } );
        // clang-format on

        mShadowSceneRenderer->Update( aWorld );
        mScene->GetNewMaterialSystem()->SetShadowMap( mShadowSceneRenderer->GetDirectionalShadowMapSampler() );
        mScene->GetNewMaterialSystem()->SetShadowMap( mShadowSceneRenderer->GetPointLightShadowMapSamplers() );
    }

    void NewSceneRenderer::Render()
    {
        SE_PROFILE_FUNCTION();

        if( !mScene )
            return;

        mShadowSceneRenderer->Render();

        auto lMaterialSystem = mScene->GetNewMaterialSystem();

        mGeometryContext->BeginRender();
        for( auto const &[_, lQueue] : mPipelines )
        {
            lMaterialSystem->SetViewParameters( mProjectionMatrix, mViewMatrix, mCameraPosition );
            lMaterialSystem->SetCameraParameters( mGamma, mExposure, mCameraPosition );

            mGeometryContext->Bind( lQueue.mPipeline );
            lMaterialSystem->ConfigureRenderContext( mGeometryContext );

            for( auto const &lMesh : lQueue.mMeshes )
            {
                lMaterialSystem->SelectMaterialInstance( mGeometryContext, lMesh.mMaterialID );

                mGeometryContext->Bind( lMesh.mVertexBuffer, lMesh.mIndexBuffer );
                mGeometryContext->Draw( lMesh.mIndexCount, lMesh.mIndexOffset, lMesh.mVertexOffset, 1, 0 );
            }
        }
        mGeometryContext->EndRender();
    }

    Ref<ITexture2D> NewSceneRenderer::GetOutputImage()
    {
        return mGeometryRenderTarget->GetAttachment( "OUTPUT" );
    }
} // namespace SE::Core