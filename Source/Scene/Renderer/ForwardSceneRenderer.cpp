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

        lCreateInfo.Opaque         = ( aPipelineSpecification.Type == eMaterialType::Opaque );
        lCreateInfo.IsTwoSided     = aPipelineSpecification.IsTwoSided;
        lCreateInfo.LineWidth      = aPipelineSpecification.LineWidth;
        lCreateInfo.VertexShader   = "Shaders\\PBRMeshShader.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\PBRMeshShader.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext.GetRenderPass();

        return lCreateInfo;
    }

    MeshRendererCreateInfo ForwardSceneRenderer::GetRenderPipelineCreateInfo( sMeshRenderData &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo;

        lCreateInfo.Opaque         = aPipelineSpecification.mOpaque;
        lCreateInfo.IsTwoSided     = aPipelineSpecification.mIsTwoSided;
        lCreateInfo.LineWidth      = aPipelineSpecification.mLineWidth;
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

    MeshRenderer &ForwardSceneRenderer::GetRenderPipeline( sMeshRenderData &aPipelineSpecification )
    {
        MeshRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    ParticleSystemRenderer &ForwardSceneRenderer::GetRenderPipeline( ParticleRendererCreateInfo &aPipelineSpecification )
    {
        if( mParticleRenderers.find( aPipelineSpecification ) == mParticleRenderers.end() )
            mParticleRenderers[aPipelineSpecification] =
                ParticleSystemRenderer( mGraphicContext, mGeometryContext, aPipelineSpecification );

        return mParticleRenderers[aPipelineSpecification];
    }

    ParticleSystemRenderer &ForwardSceneRenderer::GetRenderPipeline( sParticleShaderComponent &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
    }

    ParticleSystemRenderer &ForwardSceneRenderer::GetRenderPipeline( sParticleRenderData &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo = GetRenderPipelineCreateInfo( aPipelineSpecification );

        return GetRenderPipeline( lCreateInfo );
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

    ParticleRendererCreateInfo ForwardSceneRenderer::GetRenderPipelineCreateInfo( sParticleRenderData &aPipelineSpecification )
    {
        ParticleRendererCreateInfo lCreateInfo;
        lCreateInfo.LineWidth      = aPipelineSpecification.mLineWidth;
        lCreateInfo.VertexShader   = "Shaders\\ParticleSystem.vert.spv";
        lCreateInfo.FragmentShader = "Shaders\\ParticleSystem.frag.spv";
        lCreateInfo.RenderPass     = mGeometryContext.GetRenderPass();

        return lCreateInfo;
    }

    void ForwardSceneRenderer::Update( Ref<Scene> aWorld )
    {
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
        Settings.RenderGrayscale       = GrayscaleRendering ? 1.0f : 0.0f;

        View.Projection     = mProjectionMatrix;
        View.CameraPosition = mCameraPosition;

        mCameraUniformBuffer->Write( View );
        mShaderParametersBuffer->Write( Settings );

        // Update pipelines
    }

    void ForwardSceneRenderer::Render()
    {
        SE_PROFILE_FUNCTION();

        if( !mScene ) return;

        mScene->GetMaterialSystem()->UpdateDescriptors();

        mGeometryContext.BeginRender();
        for( auto &lPipelineData : mOpaqueMeshQueue )
        {
            auto &lPipeline = GetRenderPipeline( lPipelineData );
            if( lPipeline.Pipeline )
                mGeometryContext.Bind( lPipeline.Pipeline );
            else
                continue;
            mGeometryContext.Bind( mSceneDescriptors, 0, -1 );
            mGeometryContext.Bind( mScene->GetMaterialSystem()->GetDescriptorSet(), 1, -1 );

            if( !lPipelineData.mVertexBuffer || !lPipelineData.mIndexBuffer ) continue;
            mGeometryContext.Bind( lPipelineData.mVertexBuffer, lPipelineData.mIndexBuffer );

            MeshRenderer::MaterialPushConstants lMaterialPushConstants{};
            lMaterialPushConstants.mMaterialID = lPipelineData.mMaterialID;

            mGeometryContext.PushConstants( { eShaderStageTypeFlags::FRAGMENT }, 0, lMaterialPushConstants );

            mGeometryContext.Draw( lPipelineData.mIndexCount, lPipelineData.mIndexOffset, lPipelineData.mVertexOffset, 1, 0 );
        }

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

        if( RenderCoordinateGrid ) mCoordinateGridRenderer->Render( View.Projection, View.View, mGeometryContext );
        mGeometryContext.EndRender();
    }

    Ref<VkTexture2D> ForwardSceneRenderer::GetOutputImage()
    {
        //
        return mGeometryRenderTarget->GetAttachment( "OUTPUT" );
    }
} // namespace SE::Core