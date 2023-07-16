#include "TextureManager.h"
#include "Core/Logging.h"

#include "Scene/Serialize/AssetFile.h"

namespace SE::Core
{
    TextureManager::TextureManager( Ref<IGraphicContext> aGraphicContext )
        : mGraphicContext{ aGraphicContext }
        , mDirty{ true }
    {
        Clear();

        mTextureDescriptorLayout = CreateDescriptorSetLayout( mGraphicContext, true );
        mTextureDescriptorLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        mTextureDescriptorLayout->Build();
        mTextureDescriptorSet = mTextureDescriptorLayout->Allocate( 1024 );
    }

    void TextureManager::Clear()
    {
        mCudaTextureBuffer.Dispose();
        mTextureSamplers.clear();
    }

    uint64_t TextureManager::CreateTexture( fs::path aFilePath, sTextureSamplingInfo aSamplingInfo )
    {
        sTextureCreateInfo lTextureCreateInfo{};

        TextureData2D    lTextureImage( lTextureCreateInfo, aFilePath );
        TextureSampler2D lTextureSampler( lTextureImage, aSamplingInfo );

        return CreateTexture( lTextureImage, lTextureSampler );
    }

    uint64_t TextureManager::CreateTexture( Ref<TextureData2D> aTexture, Ref<TextureSampler2D> aTextureSampler )
    {
        if( !aTexture || !aTextureSampler ) return 0;

        return CreateTexture( *aTexture, *aTextureSampler );
    }

    uint64_t TextureManager::CreateTexture( TextureData2D &aTexture, TextureSampler2D &aTextureSampler )
    {
        auto lNewInteropTexture = CreateTexture2D( mGraphicContext, aTexture, 1, false, false, true );
        auto lNewInteropSampler = CreateSampler2D( mGraphicContext, lNewInteropTexture, aTextureSampler.mSamplingSpec );

        uint64_t lID = 0;
        for (auto const& lSampler : mTextureSamplers)
        {
            if (lSampler == nullptr)
                break;

            lID++;
        }

        if (lID >= mTextureSamplers.size())
            mTextureSamplers.push_back( lNewInteropSampler );
        else
            mTextureSamplers[lID] = lNewInteropSampler;

        mDirty = true;

        return lID;
    }

    Ref<ISampler2D> TextureManager::GetTextureByID( uint64_t aID ) { return mTextureSamplers[aID]; }

    void TextureManager::UpdateDescriptors()
    {
        if( !mDirty ) return;

        if( mCudaTextureBuffer.SizeAs<Cuda::TextureSampler2D::DeviceData>() < mTextureSamplers.size() )
        {
            mCudaTextureBuffer.Dispose();
            std::vector<Cuda::TextureSampler2D::DeviceData> lTextureDeviceData{};
            for( auto const &lCudaTextureSampler : mTextureSamplers ) lTextureDeviceData.push_back( lCudaTextureSampler->mDeviceData );

            mCudaTextureBuffer = Cuda::GPUMemory::Create( lTextureDeviceData );
        }

        mTextureDescriptorSet->Write( mTextureSamplers, 0 );

        mDirty = false;
    }
} // namespace SE::Core