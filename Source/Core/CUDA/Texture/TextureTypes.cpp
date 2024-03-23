/// @file   TextureData.cpp
///
/// @brief  Implementation file for texture data
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#include <set>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_NO_THREAD_LOCALS
#include "stb_image.h"

#include "Core/Definitions.h"

#include "TextureTypes.h"

#include "Core/Logging.h"

#include <filesystem>
#include <fstream>
#include <gli/generate_mipmaps.hpp>

namespace SE::Core
{

    image_data_t LoadImageData( fs::path const &aPath )
    {
        constexpr size_t lComponentCount = 4;

        image_data_t lImageData{};
        int32_t      lActualComponentCount = 0;
        int32_t      lWidth                = 0;
        int32_t      lHeight               = 0;
        size_t       lChannelSize          = 0;

        if( stbi_is_hdr( aPath.string().c_str() ) )
        {
            lChannelSize = 4;

            float *lData = stbi_loadf( aPath.string().c_str(), &lWidth, &lHeight, &lActualComponentCount, lComponentCount );

            if( !lData )
                return image_data_t{};

            lImageData.mFormat = color_format::RGBA32_FLOAT;
            lImageData.mPixelData =
                vector_t<uint8_t>( (uint8_t *)lData, ( (uint8_t *)lData ) + ( lWidth * lHeight * sizeof( float ) ) );
        }
        else
        {
            lChannelSize = 1;

            stbi_uc *lData = stbi_load( aPath.string().c_str(), &lWidth, &lHeight, &lActualComponentCount, lComponentCount );

            if( !lData )
                return image_data_t{};

            lImageData.mFormat    = color_format::RGBA8_UNORM;
            lImageData.mPixelData = vector_t<uint8_t>( lData, lData + ( lWidth * lHeight * sizeof( float ) ) );
        }

        lImageData.mWidth    = static_cast<size_t>( lWidth );
        lImageData.mHeight   = static_cast<size_t>( lHeight );
        lImageData.mByteSize = lImageData.mWidth * lImageData.mHeight * lComponentCount * lChannelSize;
        return lImageData;
    }
} // namespace SE::Core
