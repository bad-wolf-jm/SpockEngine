#include "ForwardSceneRenderer.h"

#include <chrono>
#include <gli/gli.hpp>

#include "Core/Vulkan/VkPipeline.h"

#include "Scene/Components/VisualHelpers.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Profiling/BlockTimer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"

namespace LTSE::Core
{

    using namespace math;
    using namespace LTSE::Core::EntityComponentSystem::Components;
    using namespace LTSE::Core::Primitives;

    // ForwardSceneRenderer::ForwardSceneRenderer( Ref<Scene> aWorld, RenderContext &mGeometryContext )
    //     : mGraphicContext{ aWorld->GetGraphicContext() }
    //     , mScene{ aWorld }
    // {
    //     mSceneDescriptors = New<DescriptorSet>( mGraphicContext, MeshRenderer::GetCameraSetLayout( mGraphicContext ) );

    //     mCameraUniformBuffer =
    //         New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( WorldMatrices ) );
    //     mShaderParametersBuffer =
    //         New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( CameraSettings ) );
    //     mSceneDescriptors->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
    //     mSceneDescriptors->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

    //     CoordinateGridRendererCreateInfo lCoordinateGridRendererCreateInfo{};
    //     lCoordinateGridRendererCreateInfo.RenderPass = mGeometryContext.GetRenderPass();
    //     mCoordinateGridRenderer = New<CoordinateGridRenderer>( mGraphicContext, mGeometryContext, lCoordinateGridRendererCreateInfo
    //     ); mVisualHelperRenderer   = New<VisualHelperRenderer>( mGraphicContext, mGeometryContext.GetRenderPass() );
    // }

    ForwardSceneRenderer::ForwardSceneRenderer(
        GraphicContext aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
        : ASceneRenderer( aGraphicContext, aOutputFormat, aOutputSampleCount )
    {
        mSceneDescriptors = New<DescriptorSet>( mGraphicContext, MeshRenderer::GetCameraSetLayout( mGraphicContext ) );

        mCameraUniformBuffer =
            New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( WorldMatrices ) );
        mShaderParametersBuffer =
            New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( CameraSettings ) );
        mSceneDescriptors->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        mSceneDescriptors->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

    }

    MeshRendererCreateInfo ForwardSceneRenderer::GetRenderPipelineCreateInfo( sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo;

        lCreateInfo.Opaque         = ( aPipelineSpecification.Type == eCMaterialType::Opaque );
        lCreateInfo.IsTwoSided     = aPipelineSpecification.IsTwoSided;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
        lCreateInfo.VertexShader   = "Shaders\\PBRMeshShader.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\PBRMeshShader.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext.GetRenderPass();

        return lCreateInfo;
    }

    void ForwardSceneRenderer::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = aOutputWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = mOutputSampleCount;
        mGeometryRenderTarget          = New<ARenderTarget>( mGraphicContext, lRenderTargetSpec );

        sAttachmentDescription lAttachmentCreateInfo{};
        // lAttachmentCreateInfo.mType        = eAttachmentType::COLOR;
        // lAttachmentCreateInfo.mClearColor  = { 0.0f, 0.0f, 0.0f, 1.0f };
        lAttachmentCreateInfo.mIsSampled   = true;
        lAttachmentCreateInfo.mIsPresented = false;
        lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;

        if( mOutputSampleCount == 1 )
        {
            lAttachmentCreateInfo.mType       = eAttachmentType::COLOR;
            lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA8_UNORM;
            lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
            mGeometryRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

            lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
            lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
            mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );
        }
        else
        {
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
        }

        mGeometryRenderTarget->Finalize();

        // lAttachmentCreateInfo.mFormat = eColorFormat::RGBA8_UNORM;
        // mGeometryRenderTarget->AddAttachment( "ALBEDO", lAttachmentCreateInfo );
        // mGeometryRenderTarget->AddAttachment( "AO_METAL_ROUGH", lAttachmentCreateInfo );

        // lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
        // lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
        // mGeometryRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );
        mGeometryContext = ARenderContext( mGraphicContext, mGeometryRenderTarget );

        CoordinateGridRendererCreateInfo lCoordinateGridRendererCreateInfo{};
        lCoordinateGridRendererCreateInfo.RenderPass = mGeometryContext.GetRenderPass();
        mCoordinateGridRenderer = New<CoordinateGridRenderer>( mGraphicContext, mGeometryContext, lCoordinateGridRendererCreateInfo );
        mVisualHelperRenderer   = New<VisualHelperRenderer>( mGraphicContext, mGeometryContext.GetRenderPass() );

        // sRenderTargetDescription lLightingSpec{};
        // lLightingSpec.mWidth       = aOutputWidth;
        // lLightingSpec.mHeight      = aOutputHeight;
        // lLightingSpec.mSampleCount = mOutputSampleCount;
        // mLightingRenderTarget      = New<ARenderTarget>( mGraphicContext, lLightingSpec );

        // lAttachmentCreateInfo.mType       = eAttachmentType::COLOR;
        // lAttachmentCreateInfo.mFormat     = eColorFormat::RGBA16_FLOAT;
        // lAttachmentCreateInfo.mClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
        // mLightingRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

        // lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
        // lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
        // mLightingRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );
        // mLightingRenderTarget->Finalize();
        // mLightingContext = ARenderContext( mGraphicContext, mLightingRenderTarget );

        // DeferredLightingRendererCreateInfo mLightingRendererCI{};
        // mLightingRendererCI.RenderPass = mLightingContext.GetRenderPass();
        // mLightingRenderer              = DeferredLightingRenderer( mGraphicContext, mLightingRendererCI );

        // mLightingPassTextures->Write(
        //     New<Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "POSITION" ) ), 0 );
        // mLightingPassTextures->Write(
        //     New<Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "NORMALS" ) ), 1 );
        // mLightingPassTextures->Write(
        //     New<Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "ALBEDO" ) ), 2 );
        // mLightingPassTextures->Write(
        //     New<Texture2D>( mGraphicContext, TextureDescription{}, mGeometryRenderTarget->GetAttachment( "AO_METAL_ROUGH" ) ), 3 );
    }

    MeshRenderer &ForwardSceneRenderer::GetRenderPipeline( MeshRendererCreateInfo const &aPipelineSpecification )
    {
        if( mMeshRenderers.find( aPipelineSpecification ) == mMeshRenderers.end() )
            mMeshRenderers[aPipelineSpecification] = MeshRenderer( mGraphicContext, aPipelineSpecification );

        return mMeshRenderers[aPipelineSpecification];
    }

    MeshRenderer &ForwardSceneRenderer::GetRenderPipeline( sMaterialShaderComponent &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    ParticleSystemRenderer &ForwardSceneRenderer::GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        if( mParticleRenderers.find( lCreateInfo ) == mParticleRenderers.end() )
            mParticleRenderers[lCreateInfo] = ParticleSystemRenderer( mGraphicContext, mGeometryContext, lCreateInfo );

        return mParticleRenderers[lCreateInfo];
        // return mParticleRenderers[ParticleRendererCreateInfo{}];
    }

    ParticleRendererCreateInfo ForwardSceneRenderer::GetRenderPipelineCreateInfo( sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
        lCreateInfo.VertexShader   = "Shaders\\ParticleSystem.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\ParticleSystem.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext.GetRenderPass();

        return lCreateInfo;
    }

    void ForwardSceneRenderer::Update( Ref<Scene> aWorld )
    {
        //
        ASceneRenderer::Update( aWorld );

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

        View.Projection     = mProjectionMatrix;
        View.CameraPosition = mCameraPosition;
        View.View           = mViewMatrix;

        mCameraUniformBuffer->Write( View );
        mShaderParametersBuffer->Write( Settings );
    }

    void ForwardSceneRenderer::Render()
    {
        LTSE_PROFILE_FUNCTION();

        if (!mScene) return;
        UpdateDescriptorSets( );
        mScene->GetMaterialSystem()->UpdateDescriptors();

        // int lDirectionalLightCount = 0;
        // int lSpotlightCount        = 0;
        // int lPointLightCount       = 0;

        // mScene->ForEach<sDirectionalLightComponent>(
        //     [&]( auto aEntity, auto &aComponent )
        //     {
        //         math::mat4 lTransformMatrix = math::mat4( 1.0f );
        //         if( aEntity.Has<sTransformMatrixComponent>() ) lTransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

        //         View.DirectionalLights[lDirectionalLightCount] = DirectionalLightData( aComponent, lTransformMatrix );
        //         lDirectionalLightCount++;
        //     } );

        // mScene->ForEach<sPointLightComponent>(
        //     [&]( auto aEntity, auto &aComponent )
        //     {
        //         math::mat4 lTransformMatrix = math::mat4( 1.0f );
        //         if( aEntity.Has<sTransformMatrixComponent>() ) lTransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

        //         View.PointLights[lPointLightCount] = PointLightData( aComponent, lTransformMatrix );
        //         lPointLightCount++;
        //     } );

        // mScene->ForEach<sSpotlightComponent>(
        //     [&]( auto aEntity, auto &aComponent )
        //     {
        //         math::mat4 lTransformMatrix = math::mat4( 1.0f );
        //         if( aEntity.Has<sTransformMatrixComponent>() ) lTransformMatrix = aEntity.Get<sTransformMatrixComponent>().Matrix;

        //         View.Spotlights[lSpotlightCount] = SpotlightData( aComponent, lTransformMatrix );
        //         lSpotlightCount++;
        //     } );

        // View.PointLightCount       = lPointLightCount;
        // View.DirectionalLightCount = lDirectionalLightCount;
        // View.SpotlightCount        = lSpotlightCount;

        // if( mScene->Environment.Has<sAmbientLightingComponent>() )
        // {
        //     auto &lComponent = mScene->Environment.Get<sAmbientLightingComponent>();

        //     Settings.AmbientLightIntensity = lComponent.Intensity;
        //     Settings.AmbientLightColor     = math::vec4( lComponent.Color, 0.0 );
        // }

        // mCameraUniformBuffer->Write( View );
        // mShaderParametersBuffer->Write( Settings );

        std::unordered_map<MeshRendererCreateInfo, std::vector<Entity>, MeshRendererCreateInfoHash> lOpaqueMeshQueue{};
        mScene->ForEach<sStaticMeshComponent, sMaterialShaderComponent>(
            [&]( auto aEntity, auto &aStaticMeshComponent, auto &aMaterialData )
            {
                auto &l_PipelineCreateInfo = GetRenderPipelineCreateInfo( aMaterialData );
                if( lOpaqueMeshQueue.find( l_PipelineCreateInfo ) == lOpaqueMeshQueue.end() )
                    lOpaqueMeshQueue[l_PipelineCreateInfo] = std::vector<Entity>{};
                lOpaqueMeshQueue[l_PipelineCreateInfo].push_back( aEntity );
            } );

        mGeometryContext.BeginRender();
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

                mGeometryContext.Bind( mSceneDescriptors, 0, -1 );
                mGeometryContext.Bind( mScene->GetMaterialSystem()->GetDescriptorSet(), 1, -1 );

                for( auto &lMeshInformation : lPipelineData.second )
                {
                    if( lMeshInformation.Has<NodeDescriptorComponent>() )
                        mGeometryContext.Bind( lMeshInformation.Get<NodeDescriptorComponent>().Descriptors, 2, -1 );

                    MeshRenderer::MaterialPushConstants l_MaterialPushConstants{};
                    l_MaterialPushConstants.mMaterialID = lMeshInformation.Get<sMaterialComponent>().mMaterialID;

                    mGeometryContext.PushConstants(
                        { Graphics::Internal::eShaderStageTypeFlags::FRAGMENT }, 0, l_MaterialPushConstants );

                    auto &l_StaticMeshComponent = lMeshInformation.Get<sStaticMeshComponent>();
                    mGeometryContext.Draw( l_StaticMeshComponent.mIndexCount, l_StaticMeshComponent.mIndexOffset,
                        l_StaticMeshComponent.mVertexOffset, 1, 0 );
                }
            }
        }

        mScene->ForEach<sParticleSystemComponent, sParticleShaderComponent>(
            [&]( auto aEntity, auto &aParticleSystemComponent, auto &aParticleShaderComponent )
            {
                auto &lPipeline = GetRenderPipeline( aParticleShaderComponent );

                ParticleSystemRenderer::ParticleData lParticleData{};
                lParticleData.Model         = math::mat4( 1.0f );
                lParticleData.ParticleCount = aParticleSystemComponent.ParticleCount;
                lParticleData.ParticleSize  = aParticleSystemComponent.ParticleSize;
                lParticleData.Particles     = aParticleSystemComponent.Particles;

                lPipeline.Render( View.Projection, View.View, mGeometryContext, lParticleData );
            } );

        if( RenderGizmos )
        {
            mVisualHelperRenderer->View       = View.View;
            mVisualHelperRenderer->Projection = View.Projection;
            mScene->ForEach<DirectionalLightHelperComponent>(
                [&]( auto aEntity, auto &a_DirectionalLightHelperComponent )
                {
                    math::mat4 l_Transform = math::mat4( 1.0f );
                    if( aEntity.Has<sTransformMatrixComponent>() ) l_Transform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                    mVisualHelperRenderer->Render( l_Transform, a_DirectionalLightHelperComponent, mGeometryContext );
                } );

            mScene->ForEach<SpotlightHelperComponent>(
                [&]( auto aEntity, auto &a_SpotlightHelperComponent )
                {
                    math::mat4 l_Transform = math::mat4( 1.0f );
                    if( aEntity.Has<sTransformMatrixComponent>() ) l_Transform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                    mVisualHelperRenderer->Render( l_Transform, a_SpotlightHelperComponent, mGeometryContext );
                } );

            mScene->ForEach<PointLightHelperComponent>(
                [&]( auto aEntity, auto &a_PointLightHelperComponent )
                {
                    math::mat4 l_Transform = math::mat4( 1.0f );
                    if( aEntity.Has<sTransformMatrixComponent>() ) l_Transform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                    mVisualHelperRenderer->Render( l_Transform, a_PointLightHelperComponent, mGeometryContext );
                } );
        }

        if( RenderCoordinateGrid ) mCoordinateGridRenderer->Render( View.Projection, View.View, mGeometryContext );
        mGeometryContext.EndRender();

    }

    Ref<sVkFramebufferImage> ForwardSceneRenderer::GetOutputImage()
    {
        //
        return mGeometryRenderTarget->GetAttachment( "OUTPUT" );
    }

    void ForwardSceneRenderer::UpdateDescriptorSets()
    {
        LTSE_PROFILE_FUNCTION();

        mScene->ForEach<sTransformMatrixComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                if( !( aEntity.Has<NodeDescriptorComponent>() ) )
                {
                    auto &l_NodeDescriptor = aEntity.Add<NodeDescriptorComponent>();
                    l_NodeDescriptor.Descriptors =
                        New<DescriptorSet>( mGraphicContext, MeshRenderer::GetNodeSetLayout( mGraphicContext ) );
                    l_NodeDescriptor.UniformBuffer = New<Buffer>(
                        mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( NodeMatrixDataComponent ) );

                    l_NodeDescriptor.Descriptors->Write(
                        l_NodeDescriptor.UniformBuffer, false, 0, sizeof( NodeMatrixDataComponent ), 0 );
                }

                auto                   &l_NodeDescriptor = aEntity.Get<NodeDescriptorComponent>();
                NodeMatrixDataComponent l_NodeTransform{};
                l_NodeTransform.Transform = aComponent.Matrix;
                aEntity.IfExists<sSkeletonComponent>(
                    [&]( auto &l_SkeletonComponent )
                    {
                        l_NodeTransform.JointCount = l_SkeletonComponent.BoneCount;
                        for( uint32_t i = 0; i < l_SkeletonComponent.BoneCount; i++ )
                            l_NodeTransform.Joints[i] = l_SkeletonComponent.JointMatrices[i];
                    } );

                l_NodeDescriptor.UniformBuffer->Write( l_NodeTransform );
            } );
    }

} // namespace LTSE::Core