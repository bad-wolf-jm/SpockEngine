#include "DeferredRenderer.h"

namespace LTSE::Core
{
    using namespace Graphics;

    DeferredRenderer::DeferredRenderer( GraphicContext aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount )
        : ASceneRenderer( aGraphicContext, aOutputFormat, mOutputSampleCount )
    {
        DescriptorSetLayoutCreateInfo l_CameraBindLayout{};
        l_CameraBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } } };

        mLightingCameraLayout = New<DescriptorSetLayout>( mGraphicContext, l_CameraBindLayout );
        mLightingPassCamera   = New<DescriptorSet>( mGraphicContext, mLightingCameraLayout );

        DescriptorSetLayoutCreateInfo lTextureBindLayout{};
        lTextureBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 2, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 3, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

        mLightingTextureLayout = New<DescriptorSetLayout>( mGraphicContext, lTextureBindLayout, false );
        mLightingPassTextures  = New<DescriptorSet>( mGraphicContext, mLightingTextureLayout );
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

    void DeferredRenderer::Update( Ref<Scene> aWorld )
    {
        //
        ASceneRenderer::Update( aWorld );
    }

    void DeferredRenderer::Render()
    {
        //
        ASceneRenderer::Render();
    }
} // namespace LTSE::Core