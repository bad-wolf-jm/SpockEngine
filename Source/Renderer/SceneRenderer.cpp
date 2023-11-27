#include "SceneRenderer.h"

#include "Core/Logging.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

#include "Common/LightInputData.hpp"
#include "Scene/Scene.h"

namespace SE::Core
{
    using namespace SE::Core::EntityComponentSystem::Components;
    using namespace SE::Core::Primitives;
    using namespace math;

    SceneRenderer::SceneRenderer( ref_t<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
        : BaseSceneRenderer( aGraphicContext, aOutputFormat, aOutputSampleCount )
    {
    }

    void SceneRenderer::CreateRenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = aOutputWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = mOutputSampleCount;

        mGeometryRenderTarget = Graphics::CreateRenderTarget( mGraphicContext, lRenderTargetSpec );
        mGeometryRenderTarget->AddAttachment( "OUTPUT",
                                              sAttachmentDescription( eAttachmentType::COLOR, eColorFormat::RGBA16_FLOAT, true ) );
        mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL",
                                              sAttachmentDescription( eAttachmentType::DEPTH, eColorFormat::UNDEFINED ) );
        mGeometryRenderTarget->Finalize();
        mGeometryContext = CreateRenderContext( mGraphicContext, mGeometryRenderTarget );
    }

    void SceneRenderer::CreateMSAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = aOutputWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = mOutputSampleCount;
        mGeometryRenderTarget          = Graphics::CreateRenderTarget( mGraphicContext, lRenderTargetSpec );

        mGeometryRenderTarget->AddAttachment( "MSAA_OUTPUT",
                                              sAttachmentDescription( eAttachmentType::COLOR, eColorFormat::RGBA16_FLOAT ) );
        mGeometryRenderTarget->AddAttachment(
            "OUTPUT", sAttachmentDescription( eAttachmentType::MSAA_RESOLVE, eColorFormat::RGBA16_FLOAT, true ) );
        mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL",
                                              sAttachmentDescription( eAttachmentType::DEPTH, eColorFormat::UNDEFINED ) );

        mGeometryRenderTarget->Finalize();
        mGeometryContext = CreateRenderContext( mGraphicContext, mGeometryRenderTarget );
    }

    void SceneRenderer::CreateFXAARenderTarget( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lFxaaSpec{};
        lFxaaSpec.mWidth       = aOutputWidth;
        lFxaaSpec.mHeight      = aOutputHeight;
        lFxaaSpec.mSampleCount = mOutputSampleCount;

        mFxaaRenderTarget = Graphics::CreateRenderTarget( mGraphicContext, lFxaaSpec );
        mFxaaRenderTarget->AddAttachment( "OUTPUT",
                                          sAttachmentDescription( eAttachmentType::COLOR, eColorFormat::RGBA16_FLOAT, true ) );
        mFxaaRenderTarget->Finalize();

        mFxaaContext = CreateRenderContext( mGraphicContext, mFxaaRenderTarget );
    }

    void SceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        if( mOutputSampleCount == 1 )
            CreateRenderTarget( aOutputWidth, aOutputHeight );
        else
            CreateMSAARenderTarget( aOutputWidth, aOutputHeight );
        mShadowSceneRenderer    = New<ShadowSceneRenderer>( mGraphicContext );
        mCoordinateGridRenderer = New<CoordinateGridRenderer>( mGraphicContext, mGeometryContext );
    }

    void SceneRenderer::Update( ref_t<Scene> aWorld )
    {
        SE_PROFILE_FUNCTION();

        BaseSceneRenderer::Update( aWorld );

        if( mScene == nullptr )
            return;

        bool                     lFoundDirectionalLight = false;
        sDirectionalLight        lDirectionalLight{};
        vector_t<sPunctualLight> lPointLights;

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

                lDirectionalLight.mColorIntensity = vec4(aComponent.mColor, aComponent.mIntensity);
                lDirectionalLight.mDirection = mat3(mScene->GetFinalTransformMatrix( aEntity ) ) * vec3{ 0.0f, 0.0f, 1.0f };
                lDirectionalLight.mCastsShadows = 1;
                
                mat4  lClip = make_mat4x4( aEntries );
                mat4 lProjection =
                    Orthogonal( vec2{ -10.0f, 10.0f }, vec2{ -10.0f, 10.0f }, vec2{ -10.0f, 10.0f } );
                mat4 lView =
                    LookAt( lDirectionalLight.mDirection * 5.0f, vec3{ 0.0f, 0.0f, 0.0f }, vec3{ 0.0f, 1.0f, 0.0f } );
                lDirectionalLight.mTransform = lClip * lProjection * lView;

                lFoundDirectionalLight = true;
                break;
            case eLightType::POINT_LIGHT:
                if( ( !aComponent.mIsOn ) )
                    return;
                auto &lNewPointLight           = lPointLights.emplace_back();
                lNewPointLight.mColorIntensity = vec4( aComponent.mColor, aComponent.mIntensity );
                lNewPointLight.mPosition       = vec3( mScene->GetFinalTransformMatrix( aEntity )[3] );
                lNewPointLight.mCastsShadows   = 1;
                break;
                }
            } );
        // clang-format on

        mScene->GetMaterialSystem()->SetLights( lDirectionalLight );
        mScene->GetMaterialSystem()->SetLights( lPointLights );

        mShadowSceneRenderer->SetLights( lDirectionalLight );
        mShadowSceneRenderer->SetLights( lPointLights );

        for( auto &[lMaterialHash, lRenderQueue] : mPipelines )
            lRenderQueue.mMeshes.clear();

        mScene->ForEach<sStaticMeshComponent, sMaterialComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterial )
            {
                size_t lMaterialHash = mScene->GetMaterialSystem()->GetMaterialHash( aMaterial.mMaterialID );
                if( mPipelines.find( lMaterialHash ) == mPipelines.end() )
                {
                    mPipelines.emplace( lMaterialHash, sRenderQueue{} );
                    mPipelines[lMaterialHash].mPipeline =
                        mScene->GetMaterialSystem()->CreateGraphicsPipeline( aMaterial.mMaterialID, mGeometryContext );
                }
                mPipelines[lMaterialHash].mMeshes.emplace_back( aStaticMeshComponent, aMaterial );
            } );
        // clang-format on

        mShadowSceneRenderer->Update( aWorld );
        mScene->GetMaterialSystem()->SetShadowMap( mShadowSceneRenderer->GetDirectionalShadowMapSampler() );
        mScene->GetMaterialSystem()->SetShadowMap( mShadowSceneRenderer->GetPointLightShadowMapSamplers() );
    }

    void SceneRenderer::Render()
    {
        SE_PROFILE_FUNCTION();

        if( !mScene )
            return;

        mShadowSceneRenderer->Render();

        auto lMaterialSystem = mScene->GetMaterialSystem();
        lMaterialSystem->SetViewParameters( mProjectionMatrix, mViewMatrix, mCameraPosition );
        lMaterialSystem->SetCameraParameters( mGamma, mExposure, mCameraPosition );

        mGeometryContext->BeginRender();
        for( auto const &[_, lQueue] : mPipelines )
        {

            mGeometryContext->Bind( lQueue.mPipeline );
            lMaterialSystem->ConfigureRenderContext( mGeometryContext );

            for( auto const &lMesh : lQueue.mMeshes )
            {
                lMaterialSystem->SelectMaterialInstance( mGeometryContext, lMesh.mMaterialID );

                mGeometryContext->Bind( lMesh.mVertexBuffer, lMesh.mIndexBuffer );
                mGeometryContext->Draw( lMesh.mIndexCount, lMesh.mIndexOffset, lMesh.mVertexOffset, 1, 0 );
            }
        }

        if( mRenderCoordinateGrid )
            mCoordinateGridRenderer->Render( mProjectionMatrix, mViewMatrix, mGeometryContext );

        mGeometryContext->EndRender();
    }

    ref_t<ITexture2D> SceneRenderer::GetOutputImage()
    {
        return mGeometryRenderTarget->GetAttachment( "OUTPUT" );
    }
} // namespace SE::Core