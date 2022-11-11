#include "DeferredRenderer.h"
#include "Core/Profiling/BlockTimer.h"

namespace LTSE::Core
{
    using namespace Graphics;

    DeferredRenderer::DeferredRenderer( GraphicContext aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
        : ASceneRenderer( aGraphicContext, aOutputFormat, mOutputSampleCount )
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

        DeferredLightingRendererCreateInfo mLightingRendererCI{};
        mLightingRendererCI.RenderPass = mLightingContext.GetRenderPass();
        mLightingRenderer              = DeferredLightingRenderer( mGraphicContext, mLightingRendererCI );

        mLightingCameraLayout = DeferredLightingRenderer::GetCameraSetLayout( mGraphicContext );
        mLightingPassCamera = New<DescriptorSet>( mGraphicContext, mLightingCameraLayout );
        mLightingPassCamera->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        mLightingPassCamera->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        mLightingTextureLayout = DeferredLightingRenderer::GetTextureSetLayout( mGraphicContext );
        mLightingPassTextures = New<DescriptorSet>( mGraphicContext, mLightingTextureLayout );
    }

    void DeferredRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lGeometrySpec{};
        lGeometrySpec.mWidth       = aOutputHeight;
        lGeometrySpec.mHeight      = aOutputHeight;
        lGeometrySpec.mSampleCount = mOutputSampleCount;
        mGeometryRenderTarget      = New<ARenderTarget>( mGraphicContext, lGeometrySpec );

        sAttachmentDescription lAttachmentCreateInfo{};
        lAttachmentCreateInfo.mType        = eAttachmentType::COLOR;
        lAttachmentCreateInfo.mClearColor  = { 0.0f, 0.0f, 0.0f, 0.0f };
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

        lAttachmentCreateInfo.mType = eAttachmentType::DEPTH;
        mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );
        mGeometryRenderTarget->Finalize();
        mGeometryContext = ARenderContext( mGraphicContext, mGeometryRenderTarget );

        sRenderTargetDescription lLightingSpec{};
        lLightingSpec.mWidth       = aOutputHeight;
        lLightingSpec.mHeight      = aOutputHeight;
        lLightingSpec.mSampleCount = mOutputSampleCount;
        mLightingRenderTarget      = New<ARenderTarget>( mGraphicContext, lLightingSpec );

        lAttachmentCreateInfo.mFormat = eColorFormat::RGBA16_FLOAT;
        mLightingRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mType = eAttachmentType::DEPTH;
        mLightingRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );
        mLightingRenderTarget->Finalize();
        mLightingContext = ARenderContext( mGraphicContext, mLightingRenderTarget );

        mLightingPassTextures->Write(
            New<Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "POSITION" ) ), 0 );
        mLightingPassTextures->Write(
            New<Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "NORMALS" ) ), 1 );
        mLightingPassTextures->Write(
            New<Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "ALBEDO" ) ), 2 );
        mLightingPassTextures->Write(
            New<Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "AO_METAL_ROUGH" ) ), 3 );
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

    void DeferredRenderer::Update( Ref<Scene> aWorld )
    {
        //
        ASceneRenderer::Update( aWorld );
    }

    void DeferredRenderer::Render()
    {
        ASceneRenderer::Render();

        // Geometry pass
        UpdateDescriptorSets();
        mScene->GetMaterialSystem()->UpdateDescriptors();

        View.PointLightCount = mPointLights.size();
        for( uint32_t i = 0; i < View.PointLightCount; i++ ) View.PointLights[i] = mPointLights[i];

        View.DirectionalLightCount = mDirectionalLights.size();
        for( uint32_t i = 0; i < View.DirectionalLightCount; i++ ) View.DirectionalLights[i] = mDirectionalLights[i];

        View.SpotlightCount = mSpotlights.size();
        for( uint32_t i = 0; i < View.SpotlightCount; i++ ) View.Spotlights[i] = mSpotlights[i];

        Settings.AmbientLightIntensity = mAmbientLight.a;
        Settings.AmbientLightColor     = math::vec4( math::vec3(mAmbientLight), 0.0 );

        mCameraUniformBuffer->Write( View );
        mShaderParametersBuffer->Write( Settings );

        std::unordered_map<MeshRendererCreateInfo, std::vector<Entity>, MeshRendererCreateInfoHash> lOpaqueMeshQueue{};
        mScene->ForEach<sStaticMeshComponent, sMaterialShaderComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterialData )
            {
                auto &lPipelineCreateInfo = GetRenderPipelineCreateInfo( aMaterialData );
                if( lOpaqueMeshQueue.find( lPipelineCreateInfo ) == lOpaqueMeshQueue.end() )
                    lOpaqueMeshQueue[lPipelineCreateInfo] = std::vector<Entity>{};
                lOpaqueMeshQueue[lPipelineCreateInfo].push_back( aEntity );
            } );

        if( mScene->mVertexBuffer && mScene->mIndexBuffer )
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

                    mGeometryContext.PushConstants(
                        { Graphics::Internal::eShaderStageTypeFlags::FRAGMENT }, 0, lMaterialPushConstants );

                    auto &lStaticMeshComponent = lMeshInformation.Get<sStaticMeshComponent>();
                    mGeometryContext.Draw( lStaticMeshComponent.mIndexCount, lStaticMeshComponent.mIndexOffset,
                        lStaticMeshComponent.mVertexOffset, 1, 0 );
                }
            }
        }

        // Lighting pass
        mLightingContext.Bind( mLightingRenderer.Pipeline );
        mLightingContext.Bind( mLightingPassCamera, 0, -1 );
        mLightingContext.Bind( mLightingPassTextures, 1, -1 );
        mLightingContext.Draw( 6, 0, 0, 1, 0 );
    }

    void DeferredRenderer::UpdateDescriptorSets()
    {
        LTSE_PROFILE_FUNCTION();

        mScene->ForEach<sTransformMatrixComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                if( !( aEntity.Has<NodeDescriptorComponent>() ) )
                {
                    auto &lNodeDescriptor = aEntity.Add<NodeDescriptorComponent>();
                    lNodeDescriptor.Descriptors =
                        New<DescriptorSet>( mGraphicContext, MeshRenderer::GetNodeSetLayout( mGraphicContext ) );
                    lNodeDescriptor.UniformBuffer = New<Buffer>(
                        mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( NodeMatrixDataComponent ) );

                    lNodeDescriptor.Descriptors->Write(
                        lNodeDescriptor.UniformBuffer, false, 0, sizeof( NodeMatrixDataComponent ), 0 );
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

} // namespace LTSE::Core