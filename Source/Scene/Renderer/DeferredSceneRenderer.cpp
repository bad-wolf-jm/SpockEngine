#include "DeferredSceneRenderer.h"
#include "Core/Profiling/BlockTimer.h"

namespace SE::Core
{
    using namespace Graphics;

    DeferredRenderer::DeferredRenderer( Ref<IGraphicContext> aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
        : ASceneRenderer( aGraphicContext, aOutputFormat, aOutputSampleCount )
    {
        // Internal uniform buffers
        mCameraUniformBuffer =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( WorldMatrices ) );
        mShaderParametersBuffer =
            CreateBuffer( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( CameraSettings ) );

        // Layout for the geometry pass
        mGeometryCameraLayout = MeshRenderer::GetCameraSetLayout( mGraphicContext );
        mGeometryPassCamera   = mGeometryCameraLayout->Allocate();
        mGeometryPassCamera->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        mGeometryPassCamera->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        mLightingCameraLayout = DeferredLightingRenderer::GetCameraSetLayout( mGraphicContext );
        mLightingPassCamera   = mLightingCameraLayout->Allocate();
        mLightingPassCamera->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        mLightingPassCamera->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        mLightingTextureLayout = DeferredLightingRenderer::GetTextureSetLayout( mGraphicContext );
        mLightingPassTextures  = mLightingTextureLayout->Allocate();

        mLightingDirectionalShadowLayout   = DeferredLightingRenderer::GetDirectionalShadowSetLayout( mGraphicContext );
        mLightingPassDirectionalShadowMaps = mLightingDirectionalShadowLayout->Allocate( 1024 );

        mLightingSpotlightShadowLayout   = DeferredLightingRenderer::GetSpotlightShadowSetLayout( mGraphicContext );
        mLightingPassSpotlightShadowMaps = mLightingSpotlightShadowLayout->Allocate( 1024 );

        mLightingPointLightShadowLayout   = DeferredLightingRenderer::GetPointLightShadowSetLayout( mGraphicContext );
        mLightingPassPointLightShadowMaps = mLightingPointLightShadowLayout->Allocate( 1024 );
    }

    void DeferredRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lGeometrySpec{};
        lGeometrySpec.mWidth       = aOutputWidth;
        lGeometrySpec.mHeight      = aOutputHeight;
        lGeometrySpec.mSampleCount = mOutputSampleCount;
        mGeometryRenderTarget      = CreateRenderTarget( mGraphicContext, lGeometrySpec );

        sAttachmentDescription lAttachmentCreateInfo{};
        lAttachmentCreateInfo.mType        = eAttachmentType::COLOR;
        lAttachmentCreateInfo.mClearColor  = { 0.0f, 0.0f, 0.0f, 1.0f };
        lAttachmentCreateInfo.mIsSampled   = true;
        lAttachmentCreateInfo.mIsPresented = false;
        lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;

        lAttachmentCreateInfo.mFormat = eColorFormat::RGBA16_FLOAT;
        mGeometryRenderTarget->AddAttachment( "POSITION", lAttachmentCreateInfo );
        mGeometryRenderTarget->AddAttachment( "NORMALS", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mFormat = eColorFormat::RGBA8_UNORM;
        mGeometryRenderTarget->AddAttachment( "ALBEDO", lAttachmentCreateInfo );
        mGeometryRenderTarget->AddAttachment( "AO_METAL_ROUGH", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mFormat = eColorFormat::R32_FLOAT;
        mGeometryRenderTarget->AddAttachment( "OBJECT_ID", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
        lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
        lAttachmentCreateInfo.mLoadOp     = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp    = eAttachmentStoreOp::STORE;
        mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );
        mGeometryRenderTarget->Finalize();
        mGeometryContext = CreateRenderContext( mGraphicContext, mGeometryRenderTarget );

        sRenderTargetDescription lLightingSpec{};
        lLightingSpec.mWidth       = aOutputWidth;
        lLightingSpec.mHeight      = aOutputHeight;
        lLightingSpec.mSampleCount = mOutputSampleCount;
        mLightingRenderTarget      = CreateRenderTarget( mGraphicContext, lLightingSpec );

        lAttachmentCreateInfo.mType       = eAttachmentType::COLOR;
        lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA16_FLOAT;
        lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
        lAttachmentCreateInfo.mLoadOp     = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp    = eAttachmentStoreOp::STORE;
        mLightingRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
        lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
        lAttachmentCreateInfo.mLoadOp     = eAttachmentLoadOp::LOAD;
        lAttachmentCreateInfo.mStoreOp    = eAttachmentStoreOp::UNSPECIFIED;
        lAttachmentCreateInfo.mIsDefined  = true;
        mLightingRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo,
                                              mGeometryRenderTarget->GetAttachment( "DEPTH_STENCIL" ) );
        mLightingRenderTarget->Finalize();
        mLightingContext = CreateRenderContext( mGraphicContext, mLightingRenderTarget );

        mLightingRenderer = New<DeferredLightingRenderer>( mGraphicContext, mLightingContext );

        mGeometrySamplers["POSITION"] = CreateSampler2D( mGraphicContext, mGeometryRenderTarget->GetAttachment( "POSITION" ) );
        mGeometrySamplers["NORMALS"]  = CreateSampler2D( mGraphicContext, mGeometryRenderTarget->GetAttachment( "NORMALS" ) );
        mGeometrySamplers["ALBEDO"]   = CreateSampler2D( mGraphicContext, mGeometryRenderTarget->GetAttachment( "ALBEDO" ) );
        mGeometrySamplers["AO_METAL_ROUGH"] =
            CreateSampler2D( mGraphicContext, mGeometryRenderTarget->GetAttachment( "AO_METAL_ROUGH" ) );

        mLightingPassTextures->Write( mGeometrySamplers["POSITION"], 0 );
        mLightingPassTextures->Write( mGeometrySamplers["NORMALS"], 1 );
        mLightingPassTextures->Write( mGeometrySamplers["ALBEDO"], 2 );
        mLightingPassTextures->Write( mGeometrySamplers["AO_METAL_ROUGH"], 3 );

        mFxaaSampler = CreateSampler2D( mGraphicContext, mLightingRenderTarget->GetAttachment( "OUTPUT" ) );

        sRenderTargetDescription lFxaaSpec{};
        lFxaaSpec.mWidth       = aOutputWidth;
        lFxaaSpec.mHeight      = aOutputHeight;
        lFxaaSpec.mSampleCount = mOutputSampleCount;
        mFxaaRenderTarget      = CreateRenderTarget( mGraphicContext, lLightingSpec );

        lAttachmentCreateInfo.mType       = eAttachmentType::COLOR;
        lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA16_FLOAT;
        lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
        lAttachmentCreateInfo.mLoadOp     = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp    = eAttachmentStoreOp::STORE;
        mFxaaRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );
        mFxaaRenderTarget->Finalize();
        mFxaaContext = CreateRenderContext( mGraphicContext, mFxaaRenderTarget );

        // CoordinateGridRendererCreateInfo lCoordinateGridRendererCreateInfo{};
        // lCoordinateGridRendererCreateInfo.RenderPass = mLightingContext->GetRenderPass();
        mCoordinateGridRenderer = New<CoordinateGridRenderer>( mGraphicContext, mLightingContext );
        mShadowSceneRenderer    = New<ShadowSceneRenderer>( mGraphicContext );

        EffectProcessorCreateInfo lEffectProcessorCreateInfo{};
        lEffectProcessorCreateInfo.mVertexShader   = "Shaders/fxaa.vert.spv";
        lEffectProcessorCreateInfo.mFragmentShader = "Shaders/fxaa.frag.spv";
        lEffectProcessorCreateInfo.RenderPass      = mFxaaContext; //->GetRenderPass();
        mFxaaRenderer                              = New<EffectProcessor>( mGraphicContext, mFxaaContext, lEffectProcessorCreateInfo );

        EffectProcessorCreateInfo lCopyCreateInfo{};
        lCopyCreateInfo.mVertexShader   = "Shaders/fxaa.vert.spv";
        lCopyCreateInfo.mFragmentShader = "Shaders/copy.frag.spv";
        lCopyCreateInfo.RenderPass      = mFxaaContext; //->GetRenderPass();
        mCopyRenderer                   = New<EffectProcessor>( mGraphicContext, mFxaaContext, lCopyCreateInfo );
    }

    Ref<ITexture2D> DeferredRenderer::GetOutputImage()
    {
        //
        return mFxaaRenderTarget->GetAttachment( "OUTPUT" );
    }

    MeshRendererCreateInfo DeferredRenderer::GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo;

        lCreateInfo.Opaque         = ( aPipelineSpecification.Type == eMaterialType::Opaque );
        lCreateInfo.IsTwoSided     = aPipelineSpecification.IsTwoSided;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
        lCreateInfo.VertexShader   = "Shaders\\Deferred\\MRT.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\Deferred\\MRT.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext;

        return lCreateInfo;
    }

    MeshRendererCreateInfo DeferredRenderer::GetRenderPipelineCreateInfo( sMeshRenderData &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo;

        lCreateInfo.Opaque         = aPipelineSpecification.mOpaque;
        lCreateInfo.IsTwoSided     = aPipelineSpecification.mIsTwoSided;
        lCreateInfo.LineWidth      = aPipelineSpecification.mLineWidth;
        lCreateInfo.VertexShader   = "Shaders\\Deferred\\MRT.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\Deferred\\MRT.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext;

        return lCreateInfo;
    }

    ParticleRendererCreateInfo DeferredRenderer::GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
        lCreateInfo.VertexShader   = "Shaders\\ParticleSystem.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\ParticleSystem.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext;

        return lCreateInfo;
    }

    ParticleRendererCreateInfo DeferredRenderer::GetRenderPipelineCreateInfo( sParticleRenderData &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo;
        lCreateInfo.LineWidth      = aPipelineSpecification.mLineWidth;
        lCreateInfo.VertexShader   = "Shaders\\ParticleSystem.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\ParticleSystem.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext;

        return lCreateInfo;
    }

    Ref<MeshRenderer> DeferredRenderer::GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification )
    {
        if( mMeshRenderers.find( aPipelineSpecification ) == mMeshRenderers.end() )
            mMeshRenderers[aPipelineSpecification] = New<MeshRenderer>( mGraphicContext, aPipelineSpecification );

        return mMeshRenderers[aPipelineSpecification];
    }

    Ref<MeshRenderer> DeferredRenderer::GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    Ref<MeshRenderer> DeferredRenderer::GetRenderPipeline( sMeshRenderData &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    Ref<ParticleSystemRenderer> DeferredRenderer::GetRenderPipeline( ParticleRendererCreateInfo &aPipelineSpecification )
    {
        if( mParticleRenderers.find( aPipelineSpecification ) == mParticleRenderers.end() )
            mParticleRenderers[aPipelineSpecification] =
                New<ParticleSystemRenderer>( mGraphicContext, mGeometryContext, aPipelineSpecification );

        return mParticleRenderers[aPipelineSpecification];
    }

    Ref<ParticleSystemRenderer> DeferredRenderer::GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    Ref<ParticleSystemRenderer> DeferredRenderer::GetRenderPipeline( sParticleRenderData &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    void DeferredRenderer::Update( Ref<Scene> aWorld )
    {
        ASceneRenderer::Update( aWorld );
        mShadowSceneRenderer->Update( aWorld );

        mView.PointLightCount = mPointLights.size();
        for( uint32_t i = 0; i < mView.PointLightCount; i++ ) mView.PointLights[i] = mPointLights[i];

        mView.DirectionalLightCount = mDirectionalLights.size();
        for( uint32_t i = 0; i < mView.DirectionalLightCount; i++ ) mView.DirectionalLights[i] = mDirectionalLights[i];

        mView.SpotlightCount = mSpotlights.size();
        for( uint32_t i = 0; i < mView.SpotlightCount; i++ ) mView.Spotlights[i] = mSpotlights[i];

        mSettings.AmbientLightIntensity = mAmbientLight.a;
        mSettings.AmbientLightColor     = math::vec4( math::vec3( mAmbientLight ), 0.0 );
        mSettings.Gamma                 = mGamma;
        mSettings.Exposure              = mExposure;
        mSettings.RenderGrayscale       = mGrayscaleRendering ? 1.0f : 0.0f;

        mView.Projection     = mProjectionMatrix;
        mView.CameraPosition = mCameraPosition;
        mView.View           = mViewMatrix;

        mCameraUniformBuffer->Write( mView );
        mShaderParametersBuffer->Write( mSettings );
    }

    void DeferredRenderer::Render()
    {
        ASceneRenderer::Render();

        // Geometry pass
        mScene->GetMaterialSystem()->UpdateDescriptors();

        mGeometryContext->BeginRender();
        for( auto &lPipelineData : mOpaqueMeshQueue )
        {
            auto &lPipeline = GetRenderPipeline( lPipelineData );
            if( lPipeline->Pipeline() )
                mGeometryContext->Bind( lPipeline->Pipeline() );
            else
                continue;
            mGeometryContext->Bind( mGeometryPassCamera, 0, -1 );
            mGeometryContext->Bind( mScene->GetMaterialSystem()->GetDescriptorSet(), 1, -1 );

            if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;
            mGeometryContext->Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );

            MeshRenderer::MaterialPushConstants lMaterialPushConstants{};
            lMaterialPushConstants.mMaterialID = lPipelineData.mMaterialID;

            mGeometryContext->PushConstants( { eShaderStageTypeFlags::FRAGMENT }, 0, lMaterialPushConstants );

            mGeometryContext->Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
        }
        mGeometryContext->EndRender();

        mShadowSceneRenderer->Render();
        if( mShadowSceneRenderer->GetDirectionalShadowMapSamplers().size() > 0 )
            mLightingPassDirectionalShadowMaps->Write( mShadowSceneRenderer->GetDirectionalShadowMapSamplers(), 0 );

        if( mShadowSceneRenderer->GetSpotlightShadowMapSamplers().size() > 0 )
            mLightingPassSpotlightShadowMaps->Write( mShadowSceneRenderer->GetSpotlightShadowMapSamplers(), 0 );

        if( mShadowSceneRenderer->GetPointLightShadowMapSamplers().size() > 0 )
            mLightingPassPointLightShadowMaps->Write( mShadowSceneRenderer->GetPointLightShadowMapSamplers(), 0 );

        // Lighting pass
        mLightingContext->BeginRender();
        {
            mLightingContext->Bind( mLightingRenderer->Pipeline() );
            mLightingContext->Bind( mLightingPassCamera, 0, -1 );
            mLightingContext->Bind( mLightingPassTextures, 1, -1 );
            mLightingContext->Bind( mLightingPassDirectionalShadowMaps, 2, -1 );
            mLightingContext->Bind( mLightingPassSpotlightShadowMaps, 3, -1 );
            mLightingContext->Bind( mLightingPassPointLightShadowMaps, 4, -1 );
            mLightingContext->Draw( 6, 0, 0, 1, 0 );

            for( auto &lParticleSystem : mParticleQueue )
            {
                auto &lPipeline = GetRenderPipeline( lParticleSystem );

                ParticleSystemRenderer::ParticleData lParticleData{};
                lParticleData.Model         = lParticleSystem.mModel;
                lParticleData.ParticleCount = lParticleSystem.mParticleCount;
                lParticleData.ParticleSize  = lParticleSystem.mParticleSize;
                lParticleData.Particles     = lParticleSystem.mParticles;

                lPipeline->Render( mView.Projection, mView.View, mGeometryContext, lParticleData );
            }

            for( auto const &lLightGizmo : mLightGizmos )
            {
                switch( lLightGizmo.mType )
                {
                case eLightType::DIRECTIONAL:
                {
                    // mVisualHelperRenderer->Render( lLightGizmo.mMatrix, aDirectionalLightHelperComponent, mGeometryContext );
                    break;
                }
                case eLightType::POINT_LIGHT:
                {
                    // mVisualHelperRenderer->Render( lLightGizmo.mMatrix, aPointLightHelperComponent, mGeometryContext );
                    break;
                }
                case eLightType::SPOTLIGHT:
                {
                    // mVisualHelperRenderer->Render( lLightGizmo.mMatrix, aSpotlightHelperComponent, mGeometryContext );
                    break;
                }
                }
            }

            if( mRenderCoordinateGrid ) mCoordinateGridRenderer->Render( mView.Projection, mView.View, mLightingContext );
        }
        mLightingContext->EndRender();

        mFxaaContext->BeginRender();
        if( mUseFXAA )
        {
            mFxaaRenderer->Render( mFxaaSampler, mFxaaContext );
        }
        else
        {
            mCopyRenderer->Render( mFxaaSampler, mFxaaContext );
        }
        mFxaaContext->EndRender();
    }
} // namespace SE::Core