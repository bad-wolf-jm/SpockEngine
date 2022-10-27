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
        NEAREST = 0, /**!< specifies nearest filtering. */
        LINEAR  = 1  /**!< specifies linear filtering. */
    };

    /** @brief Mipmap filter*/
    enum class eSamplerMipmap : uint8_t
    {
        NEAREST = 0, /**!< specifies nearest filtering. */
        LINEAR  = 1  /**!< specifies linear filtering. */
    };

    /** @brief Wrapping mode for samplers*/
    enum class eSamplerWrapping : uint32_t
    {
        REPEAT                 = 0, /**!< specifies that the repeat wrap mode will be used. */
        MIRRORED_REPEAT        = 1, /**!< specifies that the mirrored repeat  wrap mode will be used. */
        CLAMP_TO_EDGE          = 2, /**!< specifies that the clamp to edge wrap mode will be used. */
        CLAMP_TO_BORDER        = 3, /**!< specifies that the clamp to border wrap mode will be used. */
        MIRROR_CLAMP_TO_BORDER = 4  /**!< specifies that the mirror clamp to edge wrap mode will be used.  */
    };

    /** @brief Swizzles */
    enum class eSwizzleComponent : uint8_t
    {
        IDENTITY = 0, /**!< specifies that the component is set to the identity swizzle. */
        ZERO     = 1, /**!< specifies that the component is set to zero. */
        ONE      = 2, /**!< specifies that the component is set to one */
        R        = 3, /**!< specifies that the component is set to the value of the R component of the image. */
        G        = 4, /**!< specifies that the component is set to the value of the G component of the image. */
        B        = 5, /**!< specifies that the component is set to the value of the B component of the image. */
        A        = 6  /**!< specifies that the component is set to the value of the A component of the image. */
    };

    /** \struct sTextureSamplingInfo
     *
     * @brief Texture sampling configuration.
     *
     */
    struct sTextureSamplingInfo
    {
        eSamplerFilter mMinification      = eSamplerFilter::LINEAR;            //!< Specify the filter to use when the texture is being scaled up from its actual size
        eSamplerFilter mMagnification     = eSamplerFilter::LINEAR;            //!< Specify the filter to use when the texture is being scaled down from its actual size
        eSamplerMipmap mMip               = eSamplerMipmap::LINEAR;            //!< Specify the filter to use when interpolating between two elements of the mip chain
        eSamplerWrapping mWrapping        = eSamplerWrapping::CLAMP_TO_BORDER; //!< Specify the behaviour of the sampler when sampling out of bounds
        std::array<float, 2> mScaling     = { 1.0f, 1.0f };                    //!< Specified the scaling to be used.
        std::array<float, 2> mOffset      = { 0.0f, 0.0f };                    //!< Specified the offset to be used.
        std::array<float, 4> mBorderColor = { 0.0f, 0.0f, 0.0f, 0.0f };        //!< Specify the value to return when sampling out of bounds

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

        sSwizzleTransform()                      = default;
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
        eColorFormat mFormat = eColorFormat::UNDEFINED; //!< Image format
        size_t mWidth        = 0;                       //!< Width of the image, in pixels
        size_t mHeight       = 0;                       //!< Height of the image, in pixels
        size_t mByteSize     = 0;                       //!< Size of the pixel data pointer, in bytes
        uint8_t *mPixelData  = nullptr;                 //!< Raw pixel data

        template <typename PixelType> static sImageData Create( eColorFormat aFormat, size_t aWidth, size_t aHeight, uint8_t *aPixelData )
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
