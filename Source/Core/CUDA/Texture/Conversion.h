/// @file   Texture2D.h
///
/// @brief  Basic definitions for Cuda textures and samplers
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include <vector>

#include "Core/CUDA/Cuda.h"
#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/CUDA/Texture/TextureTypes.h"

#include "Core/Logging.h"
#include "Core/Definitions.h"

#include "Core/Math/Types.h"

/** \namespace SE::Cuda
 */
namespace SE::Cuda
{
    using namespace SE::Core;

    cudaChannelFormatDesc  ToCudaChannelDesc( eColorFormat aColorFormat );
    cudaTextureAddressMode ToCudaAddressMode( eSamplerWrapping aAddressMode );
    cudaTextureFilterMode  ToCudaFilterMode( eSamplerFilter aFilterMode );
} // namespace SE::Cuda
