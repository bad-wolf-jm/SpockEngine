/// @file   TextureCubeMap.cu
///
/// @brief  Implementation file for cuda textures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#include "Conversion.h"
#include "TextureCubeMap.h"

using namespace SE::Core;
namespace SE::Cuda
{
    TextureSamplerCubeMap::TextureSamplerCubeMap( Ref<Texture2D> &aTexture, const sTextureSamplingInfo &aSamplingSpec )
        : mTexture{ aTexture }
        , mSpec{ aSamplingSpec }
    {
        InitializeTextureSampler();
    }

    void TextureSamplerCubeMap::InitializeTextureSampler()
    {
        CreateTextureObject( &( mDeviceData.mTextureObject ), mTexture->mInternalCudaArray, mSpec );
    }

} // namespace SE::Cuda