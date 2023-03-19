#include "Figure.h"

namespace SE::Core
{

    Figure::Figure( Ref<VkGraphicContext> aGraphicContext )
        : mGraphicContext{ aGraphicContext }
    {
        // // Internal uniform buffers
        // mCameraUniformBuffer =
        //     New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( WorldMatrices ) );
        // mShaderParametersBuffer =
        //     New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, true, true, true, sizeof( CameraSettings ) );

        // // Layout for the geometry pass
        // mGeometryPassCamera = New<DescriptorSet>( mGraphicContext, MeshRenderer::GetCameraSetLayout( mGraphicContext ) );
        // mGeometryPassCamera->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        // mGeometryPassCamera->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        // mLightingCameraLayout = DeferredLightingRenderer::GetCameraSetLayout( mGraphicContext );
        // mLightingPassCamera   = New<DescriptorSet>( mGraphicContext, mLightingCameraLayout );
        // mLightingPassCamera->Write( mCameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        // mLightingPassCamera->Write( mShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        // mLightingTextureLayout = DeferredLightingRenderer::GetTextureSetLayout( mGraphicContext );
        // mLightingPassTextures  = New<DescriptorSet>( mGraphicContext, mLightingTextureLayout );

        // mLightingDirectionalShadowLayout   = DeferredLightingRenderer::GetDirectionalShadowSetLayout( mGraphicContext );
        // mLightingPassDirectionalShadowMaps = New<DescriptorSet>( mGraphicContext, mLightingDirectionalShadowLayout, 1024 );

        // mLightingSpotlightShadowLayout   = DeferredLightingRenderer::GetSpotlightShadowSetLayout( mGraphicContext );
        // mLightingPassSpotlightShadowMaps = New<DescriptorSet>( mGraphicContext, mLightingSpotlightShadowLayout, 1024 );

        // mLightingPointLightShadowLayout   = DeferredLightingRenderer::GetPointLightShadowSetLayout( mGraphicContext );
        // mLightingPassPointLightShadowMaps = New<DescriptorSet>( mGraphicContext, mLightingPointLightShadowLayout, 1024 );
    }

    void Figure::ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight )
    {
        mViewWidth  = aOutputWidth;
        mViewHeight = aOutputHeight;

        sRenderTargetDescription lRenderTargetSpec{};
        lRenderTargetSpec.mWidth       = mViewWidth;
        lRenderTargetSpec.mHeight      = aOutputHeight;
        lRenderTargetSpec.mSampleCount = mViewHeight;
        mRenderTarget                  = New<VkRenderTarget>( mGraphicContext, lRenderTargetSpec );

        sAttachmentDescription lAttachmentCreateInfo{};
        lAttachmentCreateInfo.mType        = eAttachmentType::COLOR;
        lAttachmentCreateInfo.mFormat      = eColorFormat::RGBA8_UNORM;
        lAttachmentCreateInfo.mClearColor  = { 0.0f, 0.0f, 0.0f, 1.0f };
        lAttachmentCreateInfo.mLoadOp      = eAttachmentLoadOp::CLEAR;
        lAttachmentCreateInfo.mStoreOp     = eAttachmentStoreOp::STORE;
        lAttachmentCreateInfo.mIsSampled   = true;
        lAttachmentCreateInfo.mIsPresented = false;
        mRenderTarget->AddAttachment( "OUTPUT", lAttachmentCreateInfo );

        lAttachmentCreateInfo.mType       = eAttachmentType::DEPTH;
        lAttachmentCreateInfo.mClearColor = { 1.0f, 0.0f, 0.0f, 0.0f };
        lAttachmentCreateInfo.mLoadOp     = eAttachmentLoadOp::LOAD;
        lAttachmentCreateInfo.mStoreOp    = eAttachmentStoreOp::UNSPECIFIED;
        lAttachmentCreateInfo.mIsDefined  = false;
        mRenderTarget->AddAttachment( "DEPTH_STENCIL", lAttachmentCreateInfo );

        mRenderTarget->Finalize();

        mRenderContext = ARenderContext( mGraphicContext, mRenderTarget );
    }

    Ref<VkTexture2D> Figure::GetOutputImage() { return mRenderTarget->GetAttachment( "OUTPUT" ); }

    void Figure::Render()
    {
        mRenderContext.BeginRender();

        mRenderContext.EndRender();
    }
} // namespace SE::Core