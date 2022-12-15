#include "MaterialSystem.h"
#include "Core/Logging.h"
namespace SE::Core
{
    MaterialSystem::MaterialSystem( Ref<VkGraphicContext> aGraphicContext )
        : mGraphicContext{ aGraphicContext }
        , mDirty{ true }
    {
        Clear();

        // The material system should be bound to a descriptor set as follows:
        // layout( set = X, binding = 1 ) readonly buffer sShaderMaterials Materials[];
        // layout( set = X, binding = 0 ) uniform sampler2D Textures[];
        DescriptorSetLayoutCreateInfo lTextureBindLayout{};
        lTextureBindLayout.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::STORAGE_BUFFER, { eShaderStageTypeFlags::FRAGMENT } },
            DescriptorBindingInfo{ 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } } };

        mShaderMaterials =
            New<VkGpuBuffer>( mGraphicContext, eBufferType::STORAGE_BUFFER, true, true, true, true, sizeof( sShaderMaterial ) );

        mTextureDescriptorLayout = New<DescriptorSetLayout>( mGraphicContext, lTextureBindLayout, true );
        mTextureDescriptorSet    = New<DescriptorSet>( mGraphicContext, mTextureDescriptorLayout, 1024 );

        mTextureDescriptorSet->Write( mShaderMaterials, false, 0, sizeof( sShaderMaterial ), 0 );
    }

    void MaterialSystem::Wipe()
    {
        mMaterials.clear();

        mCudaTextureBuffer.Dispose();
    }

    void MaterialSystem::Clear()
    {
        mMaterials.clear();
        mCudaTextureBuffer.Dispose();

        sImageData lImageDataStruct{};
        lImageDataStruct.mFormat   = eColorFormat::RGBA8_UNORM;
        lImageDataStruct.mWidth    = 1;
        lImageDataStruct.mHeight   = 1;
        lImageDataStruct.mByteSize = sizeof( uint32_t );

        sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mMipLevels = 1;

        sTextureSamplingInfo lSamplingInfo{};
        lSamplingInfo.mWrapping              = eSamplerWrapping::CLAMP_TO_EDGE;
        lSamplingInfo.mNormalizedValues      = true;
        lSamplingInfo.mNormalizedCoordinates = true;

        // Create default 1x1 black transparent texture ( tex_index: 0 )
        lImageDataStruct.mPixelData = { 0, 0, 0, 0 };

        TextureData2D    lBlackTexture( lTextureCreateInfo, lImageDataStruct );
        TextureSampler2D lBlackTextureSampler = TextureSampler2D( lBlackTexture, lSamplingInfo );
        CreateTexture( lBlackTexture, lBlackTextureSampler );

        // Create default 1x1 white texture ( tex_index: 1 )
        lImageDataStruct.mPixelData = { 255, 255, 255, 255 };
        TextureData2D    lWhiteTexture( lTextureCreateInfo, lImageDataStruct );
        TextureSampler2D lWhiteTextureSampler = TextureSampler2D( lWhiteTexture, lSamplingInfo );
        CreateTexture( lWhiteTexture, lWhiteTextureSampler );

        // Create default material (mat_index: 0)
        auto &lDefaultMaterial = CreateMaterial();

        // Default white for base color
        lDefaultMaterial.mBaseColorTexture = sTextureReference{ 0, 1 };

        // Default blask for other textures
        lDefaultMaterial.mNormalsTexture    = sTextureReference{ 0, 0 };
        lDefaultMaterial.mEmissiveTexture   = sTextureReference{ 0, 0 };
        lDefaultMaterial.mOcclusionTexture  = sTextureReference{ 0, 1 };
        lDefaultMaterial.mMetalRoughTexture = sTextureReference{ 0, 0 };
    }

    uint32_t MaterialSystem::CreateTexture( fs::path aFilePath, sTextureSamplingInfo aSamplingInfo )
    {
        sTextureCreateInfo lTextureCreateInfo{};

        TextureData2D    lTextureImage( lTextureCreateInfo, aFilePath );
        TextureSampler2D lTextureSampler( lTextureImage, aSamplingInfo );

        return CreateTexture( lTextureImage, lTextureSampler );
    }

    uint32_t MaterialSystem::CreateTexture( Ref<TextureData2D> aTexture, Ref<TextureSampler2D> aTextureSampler )
    {
        if( !aTexture || !aTextureSampler ) return 0;

        return CreateTexture( *aTexture, *aTextureSampler );
    }

    uint32_t MaterialSystem::CreateTexture( TextureData2D &aTexture, TextureSampler2D &aTextureSampler )
    {
        auto lNewInteropTexture = New<Graphics::VkTexture2D>( mGraphicContext, aTexture, 1, false, false, false );
        auto lNewInteropSampler = New<Graphics::VkSampler2D>( mGraphicContext, lNewInteropTexture, aTextureSampler.mSamplingSpec );
        mTextureSamplers.push_back( lNewInteropSampler );

        mDirty = true;

        return mTextureSamplers.size() - 1;
    }

    Ref<Graphics::VkSampler2D> MaterialSystem::GetTextureByID( uint32_t aID ) { return mTextureSamplers[aID]; }

    sMaterial &MaterialSystem::CreateMaterial()
    {
        mMaterials.emplace_back();

        mMaterials.back().mID = mMaterials.size() - 1;

        // Default white for base color
        mMaterials.back().mBaseColorTexture = sTextureReference{ 0, 1 };

        // Default blask for other textures
        mMaterials.back().mNormalsTexture    = sTextureReference{ 0, 0 };
        mMaterials.back().mEmissiveTexture   = sTextureReference{ 0, 0 };
        mMaterials.back().mOcclusionTexture  = sTextureReference{ 0, 1 };
        mMaterials.back().mMetalRoughTexture = sTextureReference{ 0, 0 };

        mDirty = true;

        return mMaterials.back();
    }

    sMaterial &MaterialSystem::CreateMaterial( sMaterial const &aMaterial )
    {
        mMaterials.push_back( aMaterial );

        mDirty = true;

        return mMaterials.back();
    }

    sMaterial &MaterialSystem::GetMaterialByID( uint32_t aID ) { return mMaterials[aID]; }

    void MaterialSystem::UpdateDescriptors()
    {
        if( !mDirty ) return;

        if( mShaderMaterials->SizeAs<sShaderMaterial>() < mMaterials.size() )
        {
            auto lBufferSize = std::max( mMaterials.size(), static_cast<size_t>( 1 ) ) * sizeof( sShaderMaterial );
            mShaderMaterials = New<VkGpuBuffer>( mGraphicContext, eBufferType::STORAGE_BUFFER, true, false, true, true, lBufferSize );
            mTextureDescriptorSet->Write( mShaderMaterials, false, 0, lBufferSize, 0 );
        }

        if( mCudaTextureBuffer.SizeAs<Cuda::TextureSampler2D::DeviceData>() < mTextureSamplers.size() )
        {
            mCudaTextureBuffer.Dispose();
            std::vector<Cuda::TextureSampler2D::DeviceData> lTextureDeviceData{};
            for( auto const &lCudaTextureSampler : mTextureSamplers ) lTextureDeviceData.push_back( lCudaTextureSampler->mDeviceData );

            mCudaTextureBuffer = Cuda::GPUMemory::Create( lTextureDeviceData );
        }

        if( mMaterials.size() > 0 )
        {
            std::vector<sShaderMaterial> lMaterialData;
            for( auto &lMat : mMaterials )
            {
                sShaderMaterial lShaderMaterial{};

                lShaderMaterial.mBaseColorFactor    = lMat.mBaseColorFactor;
                lShaderMaterial.mBaseColorUVChannel = ( lMat.mBaseColorTexture.mUVChannel == std::numeric_limits<uint32_t>::max() )
                                                          ? -1
                                                          : static_cast<int32_t>( lMat.mBaseColorTexture.mUVChannel );
                lShaderMaterial.mBaseColorTextureID = ( lMat.mBaseColorTexture.mTextureID == std::numeric_limits<uint32_t>::max() )
                                                          ? -1
                                                          : static_cast<int32_t>( lMat.mBaseColorTexture.mTextureID );

                lShaderMaterial.mMetallicFactor  = lMat.mMetallicFactor;
                lShaderMaterial.mRoughnessFactor = lMat.mRoughnessFactor;

                lShaderMaterial.mMetalnessUVChannel = lMat.mMetalRoughTexture.mUVChannel;
                lShaderMaterial.mMetalnessTextureID = lMat.mMetalRoughTexture.mTextureID;

                lShaderMaterial.mOcclusionStrength  = lMat.mOcclusionStrength;
                lShaderMaterial.mOcclusionUVChannel = lMat.mOcclusionTexture.mUVChannel;
                lShaderMaterial.mOcclusionTextureID = lMat.mOcclusionTexture.mTextureID;

                lShaderMaterial.mEmissiveFactor    = lMat.mEmissiveFactor;
                lShaderMaterial.mEmissiveUVChannel = lMat.mEmissiveTexture.mUVChannel;
                lShaderMaterial.mEmissiveTextureID = lMat.mEmissiveTexture.mTextureID;

                lShaderMaterial.mNormalUVChannel = lMat.mNormalsTexture.mUVChannel;
                lShaderMaterial.mNormalTextureID = lMat.mNormalsTexture.mTextureID;

                lShaderMaterial.mAlphaThreshold = lMat.mAlphaThreshold;

                lMaterialData.push_back( lShaderMaterial );
            }

            mShaderMaterials->Upload( lMaterialData );
        }

        mTextureDescriptorSet->Write( mTextureSamplers, 1 );

        mDirty = false;
    }
} // namespace SE::Core