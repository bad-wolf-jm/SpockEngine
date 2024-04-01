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

namespace SE::Core
{

    image_data_t LoadImageData( fs::path const &path )
    {
        constexpr size_t componentCount = 4;

        image_data_t imageData{};
        int32_t      actualComponentCount = 0;
        int32_t      width                = 0;
        int32_t      height               = 0;
        size_t       channelSize          = 0;

        if( stbi_is_hdr( path.string().c_str() ) )
        {
            channelSize = 4;

            float *data = stbi_loadf( path.string().c_str(), &width, &height, &actualComponentCount, componentCount );

            if( !data )
                return image_data_t{};

            imageData.mFormat = color_format::RGBA32_FLOAT;
            imageData.mPixelData =
                vector_t<uint8_t>( (uint8_t *)data, ( (uint8_t *)data ) + ( width * height * sizeof( float ) ) );
        }
        else
        {
            channelSize = 1;

            stbi_uc *data = stbi_load( path.string().c_str(), &width, &height, &actualComponentCount, componentCount );

            if( !data )
                return image_data_t{};

            imageData.mFormat    = color_format::RGBA8_UNORM;
            imageData.mPixelData = vector_t<uint8_t>( data, data + ( width * height * sizeof( float ) ) );
        }

        imageData.mWidth    = static_cast<size_t>( width );
        imageData.mHeight   = static_cast<size_t>( height );
        imageData.mByteSize = imageData.mWidth * imageData.mHeight * componentCount * channelSize;
        
        return imageData;
    }
} // namespace SE::Core
