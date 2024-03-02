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
    texture_sampler_cubemap_t::texture_sampler_cubemap_t( ref_t<texture2d_t> &aTexture, const texture_sampling_info_t &aSamplingSpec )
        : mTexture{ aTexture }
        , mSpec{ aSamplingSpec }
    {
        InitializeTextureSampler();
    }

    void texture_sampler_cubemap_t::InitializeTextureSampler()
    {
        CreateTextureObject( &( mDeviceData.mTextureObject ), mTexture->mInternalCudaArray, mSpec );
    }

} // namespace SE::Cuda