/// @file   Texture2D.cu
///
/// @brief  Implementation file for cuda textures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#include "Conversion.h"

#include "Core/CUDA/CudaAssert.h"

using namespace SE::Core;
namespace SE::Cuda
{
    /// @brief Convert our internal color format into a CUDA channel description
    cudaChannelFormatDesc ToCudaChannelDesc( eColorFormat aColorFormat )
    {
#ifdef CUDA_INTEROP
        switch( aColorFormat )
        {
        case eColorFormat::R32_FLOAT: return cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
        case eColorFormat::RG32_FLOAT: return cudaCreateChannelDesc( 32, 32, 0, 0, cudaChannelFormatKindFloat );
        case eColorFormat::RGB32_FLOAT: return cudaCreateChannelDesc( 32, 32, 32, 0, cudaChannelFormatKindFloat );
        case eColorFormat::RGBA32_FLOAT: return cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindFloat );
        case eColorFormat::R8_UNORM: return cudaCreateChannelDesc( 8, 0, 0, 0, cudaChannelFormatKindUnsigned );
        case eColorFormat::RG8_UNORM: return cudaCreateChannelDesc( 8, 8, 0, 0, cudaChannelFormatKindUnsigned );
        case eColorFormat::RGB8_UNORM: return cudaCreateChannelDesc( 8, 8, 8, 0, cudaChannelFormatKindUnsigned );
        case eColorFormat::RGBA8_UNORM:
        default: return cudaCreateChannelDesc( 8, 8, 8, 8, cudaChannelFormatKindUnsigned );
        }
#else
        return 0;
#endif
    }

    /// @brief Convert our wrapping descriptor into a CUDA wrapping descriptor
    cudaTextureAddressMode ToCudaAddressMode( eSamplerWrapping aAddressMode )
    {
#ifdef CUDA_INTEROP
        switch( aAddressMode )
        {
        case eSamplerWrapping::REPEAT: return cudaAddressModeWrap;
        case eSamplerWrapping::MIRRORED_REPEAT: return cudaAddressModeMirror;
        case eSamplerWrapping::CLAMP_TO_EDGE: return cudaAddressModeClamp;
        case eSamplerWrapping::CLAMP_TO_BORDER:
        case eSamplerWrapping::MIRROR_CLAMP_TO_BORDER:
        default: return cudaAddressModeBorder;
        }
#else
        return 0;
#endif
    }

    /// @brief Convert our filtering descriptor into a CUDA filtering descriptor
    cudaTextureFilterMode ToCudaFilterMode( eSamplerFilter aFilterMode )
    {
#ifdef CUDA_INTEROP
        switch( aFilterMode )
        {
        case eSamplerFilter::NEAREST: return cudaFilterModePoint;
        case eSamplerFilter::LINEAR:
        default: return cudaFilterModeLinear;
        }
#else
        return 0;
#endif
    }
} // namespace SE::Cuda