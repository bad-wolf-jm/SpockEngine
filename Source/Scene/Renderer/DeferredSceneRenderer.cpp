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
        auto lCameraDescriptorLayout = MeshRenderer::GetCameraSetLayout( mGraphicContext );
        mGeometryPassCamera          = lCameraDescriptorLayout->Allocate( 1024 );
        mGeometryPassCamera->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        mGeometryPassCamera->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        auto lLightingCameraLayout = DeferredLightingRenderer::GetCameraSetLayout( mGraphicContext );
        mLightingPassCamera        = lLightingCameraLayout->Allocate( 1 );
        mLightingPassCamera->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        mLightingPassCamera->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        auto lLightingTextureLayout = DeferredLightingRenderer::GetTextureSetLayout( mGraphicContext );
        mLightingPassTextures       = lLightingTextureLayout->Allocate();

        auto lLightingDirectionalShadowLayout = DeferredLightingRenderer::GetDirectionalShadowSetLayout( mGraphicContext );
        mLightingPassDirectionalShadowMaps    = lLightingDirectionalShadowLayout->Allocate( 1024 );

        auto lLightingSpotlightShadowLayout = DeferredLightingRenderer::GetSpotlightShadowSetLayout( mGraphicContext );
        mLightingPassSpotlightShadowMaps    = lLightingSpotlightShadowLayout->Allocate( 1024 );

        auto lLightingPointLightShadowLayout = DeferredLightingRenderer::GetPointLightShadowSetLayout( mGraphicContext );
        mLightingPassPointLightShadowMaps    = lLightingPointLightShadowLayout->Allocate( 1024 );
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
        mGeometryContext = VkRenderContext( mGraphicContext, mGeometryRenderTarget );

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

        DeferredLightingRendererCreateInfo mLightingRendererCI{};
        mLightingRendererCI.RenderPass = mLightingContext.GetRenderPass();
        mLightingRenderer              = DeferredLightingRenderer( mGraphicContext, mLightingRendererCI );

        mGeometrySamplers["POSITION"] =
            CreateSampler2D( mGraphicContext, mGeometryRenderTarget->GetAttachment( "POSITION" ), sTextureSamplingInfo{} );
        mLightingPassTextures->Write( mGeometrySamplers["POSITION"], 0 );

        mGeometrySamplers["NORMALS"] =
            CreateSampler2D( mGraphicContext, mGeometryRenderTarget->GetAttachment( "NORMALS" ), sTextureSamplingInfo{} );
        mLightingPassTextures->Write( mGeometrySamplers["NORMALS"], 1 );

        mGeometrySamplers["ALBEDO"] =
            CreateSampler2D( mGraphicContext, mGeometryRenderTarget->GetAttachment( "ALBEDO" ), sTextureSamplingInfo{} );
        mLightingPassTextures->Write( mGeometrySamplers["ALBEDO"], 2 );

        mGeometrySamplers["AO_METAL_ROUGH"] =
            CreateSampler2D( mGraphicContext, mGeometryRenderTarget->GetAttachment( "AO_METAL_ROUGH" ), sTextureSamplingInfo{} );
        mLightingPassTextures->Write( mGeometrySamplers["AO_METAL_ROUGH"], 3 );

        mFxaaSampler = CreateSampler2D( mGraphicContext, mLightingRenderTarget->GetAttachment( "OUTPUT" ), sTextureSamplingInfo{} );

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

        CoordinateGridRendererCreateInfo lCoordinateGridRendererCreateInfo{};
        lCoordinateGridRendererCreateInfo.RenderPass = mLightingContext.GetRenderPass();
        mCoordinateGridRenderer = New<CoordinateGridRenderer>( mGraphicContext, mLightingContext, lCoordinateGridRendererCreateInfo );
        mShadowSceneRenderer    = New<ShadowSceneRenderer>( mGraphicContext );

        EffectProcessorCreateInfo lEffectProcessorCreateInfo{};
        lEffectProcessorCreateInfo.mVertexShader   = "Shaders/fxaa.vert.spv";
        lEffectProcessorCreateInfo.mFragmentShader = "Shaders/fxaa.frag.spv";
        lEffectProcessorCreateInfo.RenderPass      = mFxaaContext.GetRenderPass();
        mFxaaRenderer                              = New<EffectProcessor>( mGraphicContext, mFxaaContext, lEffectProcessorCreateInfo );

        EffectProcessorCreateInfo lCopyCreateInfo{};
        lCopyCreateInfo.mVertexShader   = "Shaders/fxaa.vert.spv";
        lCopyCreateInfo.mFragmentShader = "Shaders/copy.frag.spv";
        lCopyCreateInfo.RenderPass      = mFxaaContext.GetRenderPass();
        mCopyRenderer                   = New<EffectProcessor>( mGraphicContext, mFxaaContext, lCopyCreateInfo );
    }

    Ref<VkTexture2D> DeferredRenderer::GetOutputImage()
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
        lCreateInfo.RenderPass     = mGeometryContext.GetRenderPass();

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
        lCreateInfo.RenderPass     = mGeometryContext.GetRenderPass();

        return lCreateInfo;
    }

    ParticleRendererCreateInfo DeferredRenderer::GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
        lCreateInfo.VertexShader   = "Shaders\\ParticleSystem.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\ParticleSystem.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext.GetRenderPass();

        return lCreateInfo;
    }

    ParticleRendererCreateInfo DeferredRenderer::GetRenderPipelineCreateInfo( sParticleRenderData &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo;
        lCreateInfo.LineWidth      = aPipelineSpecification.mLineWidth;
        lCreateInfo.VertexShader   = "Shaders\\ParticleSystem.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\ParticleSystem.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext.GetRenderPass();

        return lCreateInfo;
    }

    MeshRenderer &DeferredRenderer::GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification )
    {
        if( mMeshRenderers.find( aPipelineSpecification ) == mMeshRenderers.end() )
            mMeshRenderers[aPipelineSpecification] = MeshRenderer( mGraphicContext, aPipelineSpecification );

        return mMeshRenderers[aPipelineSpecification];
    }

    MeshRenderer &DeferredRenderer::GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    MeshRenderer &DeferredRenderer::GetRenderPipeline( sMeshRenderData &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    ParticleSystemRenderer &DeferredRenderer::GetRenderPipeline( ParticleRendererCreateInfo &aPipelineSpecification )
    {
        if( mParticleRenderers.find( aPipelineSpecification ) == mParticleRenderers.end() )
            mParticleRenderers[aPipelineSpecification] =
                ParticleSystemRenderer( mGraphicContext, mGeometryContext, aPipelineSpecification );

        return mParticleRenderers[aPipelineSpecification];
    }

    ParticleSystemRenderer &DeferredRenderer::GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    ParticleSystemRenderer &DeferredRenderer::GetRenderPipeline( sParticleRenderData &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    void DeferredRenderer::Update( Ref<Scene> aWorld )
    {
        ASceneRenderer::Update( aWorld );
        mShadowSceneRenderer->Update( aWorld );

        View.PointLightCount = mPointLights.size();
        for( uint32_t i = 0; i < View.PointLightCount; i++ ) View.PointLights[i] = mPointLights[i];

        View.DirectionalLightCount = mDirectionalLights.size();
        for( uint32_t i = 0; i < View.DirectionalLightCount; i++ ) View.DirectionalLights[i] = mDirectionalLights[i];

        View.SpotlightCount = mSpotlights.size();
        for( uint32_t i = 0; i < View.SpotlightCount; i++ ) View.Spotlights[i] = mSpotlights[i];

        Settings.AmbientLightIntensity = mAmbientLight.a;
        Settings.AmbientLightColor     = math::vec4( math::vec3( mAmbientLight ), 0.0 );
        Settings.Gamma                 = mGamma;
        Settings.Exposure              = mExposure;
        Settings.RenderGrayscale       = mGrayscaleRendering ? 1.0f : 0.0f;

        View.Projection     = mProjectionMatrix;
        View.CameraPosition = mCameraPosition;
        View.View           = mViewMatrix;

        mCameraUniformBuffer->Write( View );
        mShaderParametersBuffer->Write( Settings );
    }

    void DeferredRenderer::Render()
    {
        ASceneRenderer::Render();

        // Geometry pass
        mScene->GetMaterialSystem()->UpdateDescriptors();

        mGeometryContext.BeginRender();
        for( auto &lPipelineData : mOpaqueMeshQueue )
        {
            auto &lPipeline = GetRenderPipeline( lPipelineData );
            if( lPipeline.Pipeline )
                mGeometryContext.Bind( lPipeline.Pipeline );
            else
                continue;
            mGeometryContext.Bind( mGeometryPassCamera, 0, -1 );
            mGeometryContext.Bind( mScene->GetMaterialSystem()->GetDescriptorSet(), 1, -1 );

            if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;
            mGeometryContext.Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );

            MeshRenderer::MaterialPushConstants lMaterialPushConstants{};
            lMaterialPushConstants.mMaterialID = lPipelineData.mMaterialID;

            mGeometryContext.PushConstants( { eShaderStageTypeFlags::FRAGMENT }, 0, lMaterialPushConstants );

            mGeometryContext.Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
        }
        mGeometryContext.EndRender();

        mShadowSceneRenderer->Render();
        if( mShadowSceneRenderer->GetDirectionalShadowMapSamplers().size() > 0 )
            mLightingPassDirectionalShadowMaps->Write( mShadowSceneRenderer->GetDirectionalShadowMapSamplers(), 0 );

        if( mShadowSceneRenderer->GetSpotlightShadowMapSamplers().size() > 0 )
            mLightingPassSpotlightShadowMaps->Write( mShadowSceneRenderer->GetSpotlightShadowMapSamplers(), 0 );

        if( mShadowSceneRenderer->GetPointLightShadowMapSamplers().size() > 0 )
            mLightingPassPointLightShadowMaps->Write( mShadowSceneRenderer->GetPointLightShadowMapSamplers(), 0 );

        // Lighting pass
        mLightingContext.BeginRender();
        {
            mLightingContext.Bind( mLightingRenderer.Pipeline );
            mLightingContext.Bind( mLightingPassCamera, 0, -1 );
            mLightingContext.Bind( mLightingPassTextures, 1, -1 );
            mLightingContext.Bind( mLightingPassDirectionalShadowMaps, 2, -1 );
            mLightingContext.Bind( mLightingPassSpotlightShadowMaps, 3, -1 );
            mLightingContext.Bind( mLightingPassPointLightShadowMaps, 4, -1 );
            mLightingContext.Draw( 6, 0, 0, 1, 0 );

            for( auto &lParticleSystem : mParticleQueue )
            {
                auto &lPipeline = GetRenderPipeline( lParticleSystem );

                ParticleSystemRenderer::ParticleData lParticleData{};
                lParticleData.Model         = lParticleSystem.mModel;
                lParticleData.ParticleCount = lParticleSystem.mParticleCount;
                lParticleData.ParticleSize  = lParticleSystem.mParticleSize;
                lParticleData.Particles     = lParticleSystem.mParticles;

                lPipeline.Render( View.Projection, View.View, mGeometryContext, lParticleData );
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

            if( mRenderCoordinateGrid ) mCoordinateGridRenderer->Render( View.Projection, View.View, mLightingContext );
        }
        mLightingContext.EndRender();

        mFxaaContext.BeginRender();
        if( mUseFXAA )
        {
            mFxaaRenderer->Render( mFxaaSampler, mFxaaContext );
        }
        else
        {
            mCopyRenderer->Render( mFxaaSampler, mFxaaContext );
        }
        mFxaaContext.EndRender();
    }
} // namespace SE::Core