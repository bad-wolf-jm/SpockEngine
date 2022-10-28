/// @file   TextureType.h
///
/// @brief  Generic types pertaining to texture data
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <array>
#include <filesystem>
#include <memory>

#include "Graphics/API/ColorFormat.h"

namespace fs = std::filesystem;

namespace LTSE::Core
{

    /** @brief Minification and magnification filters */
    enum class eSamplerFilter : uint8_t
    {
        NEAREST = 0,
        LINEAR  = 1 
    };

    /** @brief Mipmap filter*/
    enum class eSamplerMipmap : uint8_t
    {
        NEAREST = 0,
        LINEAR  = 1 
    };

    /** @brief Wrapping mode for samplers*/
    enum class eSamplerWrapping : uint8_t
    {
        REPEAT                 = 0,
        MIRRORED_REPEAT        = 1,
        CLAMP_TO_EDGE          = 2,
        CLAMP_TO_BORDER        = 3,
        MIRROR_CLAMP_TO_BORDER = 4 
    };

    /** @brief Swizzles */
    enum class eSwizzleComponent : uint8_t
    {
        IDENTITY = 0,
        ZERO     = 1,
        ONE      = 2,
        R        = 3,
        G        = 4,
        B        = 5,
        A        = 6 
    };

    /** \struct sTextureSamplingInfo
     *
     * @brief Texture sampling configuration.
     *
     */
    struct sTextureSamplingInfo
    {
        eSamplerFilter       mMinification  = eSamplerFilter::LINEAR;
        eSamplerFilter       mMagnification = eSamplerFilter::LINEAR;
        eSamplerMipmap       mMip           = eSamplerMipmap::LINEAR;
        eSamplerWrapping     mWrapping      = eSamplerWrapping::CLAMP_TO_BORDER;
        std::array<float, 2> mScaling       = { 1.0f, 1.0f };
        std::array<float, 2> mOffset        = { 0.0f, 0.0f };
        std::array<float, 4> mBorderColor   = { 0.0f, 0.0f, 0.0f, 0.0f };

        sTextureSamplingInfo()                               = default;
        sTextureSamplingInfo( const sTextureSamplingInfo & ) = default;
    };

    /** \struct sSwizzleTransform
     *
     * @brief Swizzling is the ability to create vectors by arbitrarily rearranging the components of a given vector
     *
     * This structure specifies the swizzling used to create the texture image.
     *
     */
    struct sSwizzleTransform
    {
        eSwizzleComponent mR = eSwizzleComponent::IDENTITY;
        eSwizzleComponent mG = eSwizzleComponent::IDENTITY;
        eSwizzleComponent mB = eSwizzleComponent::IDENTITY;
        eSwizzleComponent mA = eSwizzleComponent::IDENTITY;

        sSwizzleTransform()                            = default;
        sSwizzleTransform( sSwizzleTransform const & ) = default;
    };

    /** @brief */
    enum class eTextureType : uint8_t
    {
        TEXTURE_2D = 0, //!< The texture is an ordinary 2 dimensional image
        TEXTURE_3D = 1  //!< The texture is an 3 dimensional image
    };

    /** \struct sImageData
     *
     * @brief Simple 2D image data structure
     *
     * Represents raw pixel data to be loaded into a texture class.
     *
     */
    struct sImageData
    {
        eColorFormat mFormat    = eColorFormat::UNDEFINED;
        size_t       mWidth     = 0;                      
        size_t       mHeight    = 0;                      
        size_t       mByteSize  = 0;                      
        uint8_t     *mPixelData = nullptr;                

        template <typename PixelType>
        static sImageData Create( eColorFormat aFormat, size_t aWidth, size_t aHeight, uint8_t *aPixelData )
        {
            sImageData o_ImageData{};
            o_ImageData.mWidth     = aWidth;
            o_ImageData.mHeight    = aHeight;
            o_ImageData.mFormat    = aFormat;
            o_ImageData.mByteSize  = aHeight * aWidth * sizeof( PixelType );
            o_ImageData.mPixelData = aPixelData;

            return o_ImageData;
        }
    };

    /// @brief Load an image file using stb_image. The image can be in any format that stb_image can handle
    sImageData LoadImageData( fs::path const &aPath );

} // namespace LTSE::Core
