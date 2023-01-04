#include "ForwardSceneRenderer.h"

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

    ForwardSceneRenderer::ForwardSceneRenderer( Ref<VkGraphicContext> aGraphicContext, eColorFormat aOutputFormat,
                                                uint32_t aOutputSampleCount )
        : ASceneRenderer( aGraphicContext, aOutputFormat, aOutputSampleCount )
    {
        mSceneDescriptors = New<DescriptorSet>( mGraphicContext, MeshRenderer::GetCameraSetLayout( mGraphicContext ) );

        mCameraUniformBuffer =
            New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( WorldMatrices ) );
        mShaderParametersBuffer =
            New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( CameraSettings ) );
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
        mGeometryRenderTarget          = New<VkRenderTarget>( mGraphicContext, lRenderTargetSpec );

        sAttachmentDescription lAttachmentCreateInfo{};
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

        mGeometryContext = ARenderContext( mGraphicContext, mGeometryRenderTarget );

        CoordinateGridRendererCreateInfo lCoordinateGridRendererCreateInfo{};
        lCoordinateGridRendererCreateInfo.RenderPass = mGeometryContext.GetRenderPass();
        mCoordinateGridRenderer = New<CoordinateGridRenderer>( mGraphicContext, mGeometryContext, lCoordinateGridRendererCreateInfo );
        mVisualHelperRenderer   = New<VisualHelperRenderer>( mGraphicContext, mGeometryContext.GetRenderPass() );
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
        Settings.RenderGrayscale       = GrayscaleRendering ? 1.0f : 0.0f;

        View.Projection     = mProjectionMatrix;
        View.CameraPosition = mCameraPosition;

        mCameraUniformBuffer->Write( View );
        mShaderParametersBuffer->Write( Settings );
    }

    void ForwardSceneRenderer::Render()
    {
        SE_PROFILE_FUNCTION();

        if( !mScene ) return;
        UpdateDescriptorSets();
        mScene->GetMaterialSystem()->UpdateDescriptors();

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
        // if( mScene->mVertexBuffer && mScene->mIndexBuffer )
        // {
        //     mGeometryContext.Bind( mScene->mTransformedVertexBuffer, mScene->mIndexBuffer );
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
                auto &lStaticMeshData = lMeshInformation.Get<sStaticMeshComponent>();
                if( !lStaticMeshData.mTransformedBuffer || !lStaticMeshData.mIndexBuffer ) continue;
                mGeometryContext.Bind( lStaticMeshData.mTransformedBuffer, lStaticMeshData.mIndexBuffer );
                if( lMeshInformation.Has<NodeDescriptorComponent>() )
                    mGeometryContext.Bind( lMeshInformation.Get<NodeDescriptorComponent>().Descriptors, 2, -1 );

                MeshRenderer::MaterialPushConstants l_MaterialPushConstants{};
                l_MaterialPushConstants.mMaterialID = lMeshInformation.Get<sMaterialComponent>().mMaterialID;

                mGeometryContext.PushConstants( { eShaderStageTypeFlags::FRAGMENT }, 0, l_MaterialPushConstants );

                auto &l_StaticMeshComponent = lMeshInformation.Get<sStaticMeshComponent>();
                mGeometryContext.Draw( l_StaticMeshComponent.mIndexCount, l_StaticMeshComponent.mIndexOffset,
                                       l_StaticMeshComponent.mVertexOffset, 1, 0 );
            }
        }
        // }

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
                [&]( auto aEntity, auto &aDirectionalLightHelperComponent )
                {
                    math::mat4 lTransform = math::mat4( 1.0f );
                    if( aEntity.Has<sTransformMatrixComponent>() ) lTransform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                    mVisualHelperRenderer->Render( lTransform, aDirectionalLightHelperComponent, mGeometryContext );
                } );

            mScene->ForEach<SpotlightHelperComponent>(
                [&]( auto aEntity, auto &aSpotlightHelperComponent )
                {
                    math::mat4 lTransform = math::mat4( 1.0f );
                    if( aEntity.Has<sTransformMatrixComponent>() ) lTransform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                    mVisualHelperRenderer->Render( lTransform, aSpotlightHelperComponent, mGeometryContext );
                } );

            mScene->ForEach<PointLightHelperComponent>(
                [&]( auto aEntity, auto &aPointLightHelperComponent )
                {
                    math::mat4 lTransform = math::mat4( 1.0f );
                    if( aEntity.Has<sTransformMatrixComponent>() ) lTransform = aEntity.Get<sTransformMatrixComponent>().Matrix;
                    mVisualHelperRenderer->Render( lTransform, aPointLightHelperComponent, mGeometryContext );
                } );
        }

        if( RenderCoordinateGrid ) mCoordinateGridRenderer->Render( View.Projection, View.View, mGeometryContext );
        mGeometryContext.EndRender();
    }

    Ref<VkTexture2D> ForwardSceneRenderer::GetOutputImage()
    {
        //
        return mGeometryRenderTarget->GetAttachment( "OUTPUT" );
    }

    void ForwardSceneRenderer::UpdateDescriptorSets()
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
                    lNodeDescriptor.UniformBuffer = New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true,
                                                                      true, sizeof( NodeMatrixDataComponent ) );

                    lNodeDescriptor.Descriptors->Write( lNodeDescriptor.UniformBuffer, false, 0, sizeof( NodeMatrixDataComponent ),
                                                        0 );
                }

                NodeMatrixDataComponent lNodeTransform{};
                lNodeTransform.Transform = aComponent.Matrix;
                aEntity.IfExists<sSkeletonComponent>(
                    [&]( auto &l_SkeletonComponent )
                    {
                        lNodeTransform.JointCount = l_SkeletonComponent.BoneCount;
                        for( uint32_t i = 0; i < l_SkeletonComponent.BoneCount; i++ )
                            lNodeTransform.Joints[i] = l_SkeletonComponent.JointMatrices[i];
                    } );

                auto &lNodeDescriptor = aEntity.Get<NodeDescriptorComponent>();
                lNodeDescriptor.UniformBuffer->Write( lNodeTransform );
            } );
    }

} // namespace SE::Core