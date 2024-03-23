/// @file   TextureType.h
///
/// @brief  Generic types pertaining to texture data
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include <array>
#include <filesystem>
#include <memory>

#include "Core/Definitions.h"

namespace fs = std::filesystem;

namespace SE::Core
{
    enum class color_format : uint32_t
    {
        UNDEFINED           = 0,
        R32_FLOAT           = 1,
        RG32_FLOAT          = 2,
        RGB32_FLOAT         = 3,
        RGBA32_FLOAT        = 4,
        R16_FLOAT           = 5,
        RG16_FLOAT          = 6,
        RGB16_FLOAT         = 7,
        RGBA16_FLOAT        = 8,
        R8_UNORM            = 9,
        RG8_UNORM           = 10,
        RGB8_UNORM          = 11,
        RGBA8_UNORM         = 12,
        D16_UNORM           = 13,
        X8_D24_UNORM_PACK32 = 14,
        D32_SFLOAT          = 15,
        S8_UINT             = 16,
        D16_UNORM_S8_UINT   = 17,
        D24_UNORM_S8_UINT   = 18,
        D32_UNORM_S8_UINT   = 19,
        BGR8_SRGB           = 20,
        BGRA8_SRGB          = 21
    };

    /** @brief Minification and magnification filters */
    enum class sampler_filter_t : uint8_t
    {
        NEAREST = 0, /**!< specifies nearest filtering. */
        LINEAR  = 1  /**!< specifies linear filtering. */
    };

    /** @brief Mipmap filter*/
    enum class sampler_mipmap_t : uint8_t
    {
        NEAREST = 0, /**!< specifies nearest filtering. */
        LINEAR  = 1  /**!< specifies linear filtering. */
    };

    /** @brief Wrapping mode for samplers*/
    enum class sampler_wrapping_t : uint32_t
    {
        REPEAT                 = 0, /**!< specifies that the repeat wrap mode will be used. */
        MIRRORED_REPEAT        = 1, /**!< specifies that the mirrored repeat  wrap mode will be used. */
        CLAMP_TO_EDGE          = 2, /**!< specifies that the clamp to edge wrap mode will be used. */
        CLAMP_TO_BORDER        = 3, /**!< specifies that the clamp to border wrap mode will be used. */
        MIRROR_CLAMP_TO_BORDER = 4  /**!< specifies that the mirror clamp to edge wrap mode will be used.  */
    };

    /** \struct sTextureSamplingInfo
     *
     * @brief Texture sampling configuration.
     *
     */
    struct texture_sampling_info_t
    {
        sampler_filter_t   mFilter                = sampler_filter_t::LINEAR;
        sampler_mipmap_t   mMipFilter             = sampler_mipmap_t::LINEAR;
        sampler_wrapping_t mWrapping              = sampler_wrapping_t::CLAMP_TO_BORDER;
        bool               mNormalizedCoordinates = false;
        bool               mNormalizedValues      = false;

        std::array<float, 2> mScaling     = { 1.0f, 1.0f };             //!< Specified the scaling to be used.
        std::array<float, 2> mOffset      = { 0.0f, 0.0f };             //!< Specified the offset to be used.
        std::array<float, 4> mBorderColor = { 0.0f, 0.0f, 0.0f, 0.0f }; //!< Specify the value to return when sampling out of bounds

        texture_sampling_info_t()                                  = default;
        texture_sampling_info_t( const texture_sampling_info_t & ) = default;
    };

    /** @brief */
    enum class texture_type_t : uint8_t
    {
        UNKNOWN          = 0,
        TEXTURE_2D       = 1, //!< The texture is an ordinary 2 dimensional image
        TEXTURE_3D       = 2, //!< The texture is an 3 dimensional image, //!< The texture is an ordinary 2 dimensional image
        TEXTURE_CUBE_MAP = 3  //!< The texture is an 3 dimensional image
    };

    /** \struct sImageData
     *
     * @brief Simple 2D image data structure
     *
     * Represents raw pixel data to be loaded into a texture class.
     *
     */
    struct image_data_t
    {
        color_format    mFormat    = color_format::UNDEFINED; //!< Image format
        size_t            mWidth     = 0;                         //!< Width of the image, in pixels
        size_t            mHeight    = 0;                         //!< Height of the image, in pixels
        size_t            mByteSize  = 0;                         //!< Size of the pixel data pointer, in bytes
        vector_t<uint8_t> mPixelData = {};                        //!< Raw pixel data

        template <typename PixelType>
        static image_data_t Create( color_format aFormat, size_t aWidth, size_t aHeight, uint8_t *aPixelData )
        {
            image_data_t o_ImageData{};
            o_ImageData.mWidth     = aWidth;
            o_ImageData.mHeight    = aHeight;
            o_ImageData.mFormat    = aFormat;
            o_ImageData.mByteSize  = aHeight * aWidth * sizeof( PixelType );
            o_ImageData.mPixelData = vector_t<uint8_t>( aPixelData, aPixelData + o_ImageData.mByteSize );

            return o_ImageData;
        }
    };

    /// @brief Load an image file using stb_image. The image can be in any format that stb_image can handle
    image_data_t LoadImageData( fs::path const &aPath );

    struct texture_create_info_t
    {
        texture_type_t mType =
            texture_type_t::TEXTURE_2D; /**!< Specifies the type of texture. Possible values are TEXTURE_2D and TEXTURE_3D*/

        color_format mFormat         = color_format::UNDEFINED; /**!< Specifies the color format used for the texture*/
        int32_t        mWidth          = 0;                         /**!< Width of the texture, in pixels*/
        int32_t        mHeight         = 0;                         /**!< Height of the texture, in pixels*/
        int32_t        mDepth          = 0;                         /**!< Depth of the texture, in pixels (for 3D textures only)*/
        int32_t        mMipLevels      = 1; /**!< Specifies the length of the mip chain associated with the texture*/
        int32_t        mLayers         = 1; /**!< Depth of the texture, in pixels (for 3D textures only)*/
        bool           mIsDepthTexture = false;

        texture_create_info_t()                                = default;
        texture_create_info_t( texture_create_info_t const & ) = default;
    };

    enum cube_face_t : uint8_t
    {
        POSITIVE_X = 0,
        NEGATIVE_X = 1,
        POSITIVE_Y = 2,
        NEGATIVE_Y = 3,
        POSITIVE_Z = 4,
        NEGATIVE_Z = 5
    };
} // namespace SE::Core
