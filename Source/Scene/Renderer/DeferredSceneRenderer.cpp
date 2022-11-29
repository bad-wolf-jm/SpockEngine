#include "DeferredSceneRenderer.h"
#include "Core/Profiling/BlockTimer.h"

namespace SE::Core
{
    using namespace Graphics;

    DeferredRenderer::DeferredRenderer( GraphicContext aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
        : ASceneRenderer( aGraphicContext, aOutputFormat, aOutputSampleCount )
    {
        // Internal uniform buffers
        mCameraUniformBuffer =
            New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( WorldMatrices ) );
        mShaderParametersBuffer =
            New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( CameraSettings ) );

        // Layout for the geometry pass
        mGeometryPassCamera = New<DescriptorSet>( mGraphicContext, MeshRenderer::GetCameraSetLayout( mGraphicContext ) );
        mGeometryPassCamera->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        mGeometryPassCamera->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        mLightingCameraLayout = DeferredLightingRenderer::GetCameraSetLayout( mGraphicContext );
        mLightingPassCamera   = New<DescriptorSet>( mGraphicContext, mLightingCameraLayout );
        mLightingPassCamera->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        mLightingPassCamera->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        mLightingTextureLayout = DeferredLightingRenderer::GetTextureSetLayout( mGraphicContext );
        mLightingPassTextures  = New<DescriptorSet>( mGraphicContext, mLightingTextureLayout );
    }

    void DeferredRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lGeometrySpec{};
        lGeometrySpec.mWidth       = aOutputWidth;
        lGeometrySpec.mHeight      = aOutputHeight;
        lGeometrySpec.mSampleCount = mOutputSampleCount;
        mGeometryRenderTarget      = New<ARenderTarget>( mGraphicContext, lGeometrySpec );

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
        mGeometryContext = ARenderContext( mGraphicContext, mGeometryRenderTarget );

        sRenderTargetDescription lLightingSpec{};
        lLightingSpec.mWidth       = aOutputWidth;
        lLightingSpec.mHeight      = aOutputHeight;
        lLightingSpec.mSampleCount = mOutputSampleCount;
        mLightingRenderTarget      = New<ARenderTarget>( mGraphicContext, lLightingSpec );

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
        mLightingContext = ARenderContext( mGraphicContext, mLightingRenderTarget );

        DeferredLightingRendererCreateInfo mLightingRendererCI{};
        mLightingRendererCI.RenderPass = mLightingContext.GetRenderPass();
        mLightingRenderer              = DeferredLightingRenderer( mGraphicContext, mLightingRendererCI );

        mLightingPassTextures->Write(
            New<Graphics::Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "POSITION" ) ), 0 );
        mLightingPassTextures->Write(
            New<Graphics::Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "NORMALS" ) ), 1 );
        mLightingPassTextures->Write(
            New<Graphics::Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "ALBEDO" ) ), 2 );
        mLightingPassTextures->Write( New<Graphics::Texture2D>( mGraphicContext, TextureDescription{},
                                                                mGeometryRenderTarget->GetAttachment( "AO_METAL_ROUGH" ) ),
                                      3 );

        CoordinateGridRendererCreateInfo lCoordinateGridRendererCreateInfo{};
        lCoordinateGridRendererCreateInfo.RenderPass = mLightingContext.GetRenderPass();
        mCoordinateGridRenderer = New<CoordinateGridRenderer>( mGraphicContext, mLightingContext, lCoordinateGridRendererCreateInfo );
        mVisualHelperRenderer   = New<VisualHelperRenderer>( mGraphicContext, mLightingContext.GetRenderPass() );
    }

    Ref<sVkFramebufferImage> DeferredRenderer::GetOutputImage()
    {
        //
        return mLightingRenderTarget->GetAttachment( "OUTPUT" );
    }

    MeshRendererCreateInfo DeferredRenderer::GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo;

        lCreateInfo.Opaque         = ( aPipelineSpecification.Type == eCMaterialType::Opaque );
        lCreateInfo.IsTwoSided     = aPipelineSpecification.IsTwoSided;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
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

    ParticleSystemRenderer &DeferredRenderer::GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        if( mParticleRenderers.find( lCreateInfo ) == mParticleRenderers.end() )
            mParticleRenderers[lCreateInfo] = ParticleSystemRenderer( mGraphicContext, mLightingContext, lCreateInfo );

        return mParticleRenderers[lCreateInfo];
    }

    void DeferredRenderer::Update( Ref<Scene> aWorld )
    {
        //
        ASceneRenderer::Update( aWorld );

        if( aWorld->Environment.Has<sAmbientLightingComponent>() )
        {
            auto &l_AmbientLightComponent = aWorld->Environment.Get<sAmbientLightingComponent>();
            SetAmbientLighting( math::vec4( l_AmbientLightComponent.Color, l_AmbientLightComponent.Intensity ) );
        }

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
        Settings.RenderGrayscale       = mGrayscaleRendering ? 1.0f: 0.0f;

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
        UpdateDescriptorSets();
        mScene->GetMaterialSystem()->UpdateDescriptors();

        std::unordered_map<MeshRendererCreateInfo, std::vector<Entity>, MeshRendererCreateInfoHash> lOpaqueMeshQueue{};
        mScene->ForEach<sStaticMeshComponent, sMaterialShaderComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterialData )
            {
                auto &lPipelineCreateInfo = GetRenderPipelineCreateInfo( aMaterialData );
                if( lOpaqueMeshQueue.find( lPipelineCreateInfo ) == lOpaqueMeshQueue.end() )
                    lOpaqueMeshQueue[lPipelineCreateInfo] = std::vector<Entity>{};
                lOpaqueMeshQueue[lPipelineCreateInfo].push_back( aEntity );
            } );

        mGeometryContext.BeginRender();
        if( mScene->mVertexBuffer && mScene->mIndexBuffer )
        {
            {
                mGeometryContext.Bind( mScene->mTransformedVertexBuffer, mScene->mIndexBuffer );
                for( auto &lPipelineData : lOpaqueMeshQueue )
                {
                    auto &lPipeline = GetRenderPipeline( lPipelineData.first );
                    if( lPipeline.Pipeline )
                        mGeometryContext.Bind( lPipeline.Pipeline );
                    else
                        continue;

                    mGeometryContext.Bind( mGeometryPassCamera, 0, -1 );
                    mGeometryContext.Bind( mScene->GetMaterialSystem()->GetDescriptorSet(), 1, -1 );

                    for( auto &lMeshInformation : lPipelineData.second )
                    {
                        if( lMeshInformation.Has<NodeDescriptorComponent>() )
                            mGeometryContext.Bind( lMeshInformation.Get<NodeDescriptorComponent>().Descriptors, 2, -1 );

                        MeshRenderer::MaterialPushConstants lMaterialPushConstants{};
                        lMaterialPushConstants.mMaterialID = lMeshInformation.Get<sMaterialComponent>().mMaterialID;

                        mGeometryContext.PushConstants( { Graphics::Internal::eShaderStageTypeFlags::FRAGMENT }, 0,
                                                        lMaterialPushConstants );

                        auto &lStaticMeshComponent = lMeshInformation.Get<sStaticMeshComponent>();
                        mGeometryContext.Draw( lStaticMeshComponent.mIndexCount, lStaticMeshComponent.mIndexOffset,
                                               lStaticMeshComponent.mVertexOffset, 1, 0 );
                    }
                }
            }
        }
        mGeometryContext.EndRender();

        // Lighting pass
        mLightingContext.BeginRender();
        {
            mLightingContext.Bind( mLightingRenderer.Pipeline );
            mLightingContext.Bind( mLightingPassCamera, 0, -1 );
            mLightingContext.Bind( mLightingPassTextures, 1, -1 );
            mLightingContext.Draw( 6, 0, 0, 1, 0 );

            mScene->ForEach<sParticleSystemComponent, sParticleShaderComponent>(
                [&]( auto aEntity, auto &aParticleSystemComponent, auto &aParticleShaderComponent )
                {
                    auto &lPipeline = GetRenderPipeline( aParticleShaderComponent );

                    ParticleSystemRenderer::ParticleData lParticleData{};
                    lParticleData.Model         = math::mat4( 1.0f );
                    lParticleData.ParticleCount = aParticleSystemComponent.ParticleCount;
                    lParticleData.ParticleSize  = aParticleSystemComponent.ParticleSize;
                    lParticleData.Particles     = aParticleSystemComponent.Particles;

                    lPipeline.Render( View.Projection, View.View, mLightingContext, lParticleData );
                } );

            if( mRenderGizmos )
            {
                mVisualHelperRenderer->View       = View.View;
                mVisualHelperRenderer->Projection = View.Projection;
                mScene->ForEach<DirectionalLightHelperComponent>(
                    [&]( auto aEntity, auto &aDirectionalLightHelperComponent )
                    {
                        math::mat4 lTransform = math::mat4( 1.0f );
                        if( aEntity.Has<sTransformMatrixComponent>() ) lTransform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                        mVisualHelperRenderer->Render( lTransform, aDirectionalLightHelperComponent, mLightingContext );
                    } );

                mScene->ForEach<SpotlightHelperComponent>(
                    [&]( auto aEntity, auto &aSpotlightHelperComponent )
                    {
                        math::mat4 lTransform = math::mat4( 1.0f );
                        if( aEntity.Has<sTransformMatrixComponent>() ) lTransform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                        mVisualHelperRenderer->Render( lTransform, aSpotlightHelperComponent, mLightingContext );
                    } );

                mScene->ForEach<PointLightHelperComponent>(
                    [&]( auto aEntity, auto &aPointLightHelperComponent )
                    {
                        math::mat4 lTransform = math::mat4( 1.0f );
                        if( aEntity.Has<sTransformMatrixComponent>() ) lTransform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                        mVisualHelperRenderer->Render( lTransform, aPointLightHelperComponent, mLightingContext );
                    } );
            }

            if( mRenderCoordinateGrid ) mCoordinateGridRenderer->Render( View.Projection, View.View, mLightingContext );
        }
        mLightingContext.EndRender();
    }

    void DeferredRenderer::UpdateDescriptorSets()
    {
        SE_PROFILE_FUNCTION();

        mScene->ForEach<sTransformMatrixComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                if( !( aEntity.Has<NodeDescriptorComponent>() ) )
                {
                    auto &lNodeDescriptor = aEntity.Add<NodeDescriptorComponent>();
                    lNodeDescriptor.Descriptors =
                        New<DescriptorSet>( mGraphicContext, MeshRenderer::GetNodeSetLayout( mGraphicContext ) );
                    lNodeDescriptor.UniformBuffer = New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true,
                                                                 true, sizeof( NodeMatrixDataComponent ) );

                    lNodeDescriptor.Descriptors->Write( lNodeDescriptor.UniformBuffer, false, 0, sizeof( NodeMatrixDataComponent ),
                                                        0 );
                }

                auto &lNodeDescriptor = aEntity.Get<NodeDescriptorComponent>();

                NodeMatrixDataComponent lNodeTransform{};
                lNodeTransform.Transform = aComponent.Matrix;
                aEntity.IfExists<sSkeletonComponent>(
                    [&]( auto &aSkeletonComponent )
                    {
                        lNodeTransform.JointCount = aSkeletonComponent.BoneCount;
                        for( uint32_t i = 0; i < aSkeletonComponent.BoneCount; i++ )
                            lNodeTransform.Joints[i] = aSkeletonComponent.JointMatrices[i];
                    } );

                lNodeDescriptor.UniformBuffer->Write( lNodeTransform );
            } );
    }

} // namespace SE::Core