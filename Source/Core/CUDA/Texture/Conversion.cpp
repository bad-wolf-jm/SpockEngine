/// @file   Texture2D.cu
///
/// @brief  Implementation file for cuda textures
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#include "Conversion.h"

using namespace SE::Core;
namespace SE::Cuda
{
    /// @brief Convert our internal color format into a CUDA channel description
    cudaChannelFormatDesc ToCudaChannelDesc( color_format colorFormat )
    {
        switch( colorFormat )
        {
        case color_format::R32_FLOAT:
            return cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
        case color_format::RG32_FLOAT:
            return cudaCreateChannelDesc( 32, 32, 0, 0, cudaChannelFormatKindFloat );
        case color_format::RGB32_FLOAT:
            return cudaCreateChannelDesc( 32, 32, 32, 0, cudaChannelFormatKindFloat );
        case color_format::RGBA32_FLOAT:
            return cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindFloat );
        case color_format::R8_UNORM:
            return cudaCreateChannelDesc( 8, 0, 0, 0, cudaChannelFormatKindUnsigned );
        case color_format::RG8_UNORM:
            return cudaCreateChannelDesc( 8, 8, 0, 0, cudaChannelFormatKindUnsigned );
        case color_format::RGB8_UNORM:
            return cudaCreateChannelDesc( 8, 8, 8, 0, cudaChannelFormatKindUnsigned );
        case color_format::RGBA8_UNORM:
        default:
            return cudaCreateChannelDesc( 8, 8, 8, 8, cudaChannelFormatKindUnsigned );
        }
    }

    /// @brief Convert our wrapping descriptor into a CUDA wrapping descriptor
    cudaTextureAddressMode ToCudaAddressMode( sampler_wrapping addressMode )
    {
        switch( addressMode )
        {
        case sampler_wrapping::REPEAT:
            return cudaAddressModeWrap;
        case sampler_wrapping::MIRRORED_REPEAT:
            return cudaAddressModeMirror;
        case sampler_wrapping::CLAMP_TO_EDGE:
            return cudaAddressModeClamp;
        case sampler_wrapping::CLAMP_TO_BORDER:
        case sampler_wrapping::MIRROR_CLAMP_TO_BORDER:
        default:
            return cudaAddressModeBorder;
        }
    }

    /// @brief Convert our filtering descriptor into a CUDA filtering descriptor
    cudaTextureFilterMode ToCudaFilterMode( sampler_filter filterMode )
    {
        switch( filterMode )
        {
        case sampler_filter::NEAREST:
            return cudaFilterModePoint;
        case sampler_filter::LINEAR:
        default:
            return cudaFilterModeLinear;
        }
    }
} // namespace SE::Cuda