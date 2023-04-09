/// @file   TextureData.h
///
/// @brief  CPU side texture access
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include <array>
#include <filesystem>
#include <memory>

#include <gli/gli.hpp>

#include "ColorFormat.h"
#include "TextureTypes.h"

namespace fs = std::filesystem;

namespace SE::Core
{

    /** \struct TextureData
     *
     * @brief CPU-side class for textures
     *
     */
    class TextureData
    {
      public:
        /** @brief Texture creation metadata */

      public:
        sTextureCreateInfo mSpec; /**!< Copy of the sCreateInfo structure used ot define the texture. */

        /** @brief Default constructor*/
        TextureData() = default;

        /** @brief Default destructor */
        ~TextureData() = default;

        /** @brief Construct a texture from a create information structure
         *
         * This also allocates the memory for the texture data, the amount of which is determined from the information
         * in the `aCreateInfo` structure. No initialization is performed by the constructor.
         *
         * @param aCreateInfo Creation structure
         */
        TextureData( sTextureCreateInfo const &aCreateInfo );

        /** @brief Construct a texture from a create information structure and initial data
         *
         * Note that when this version of the constructor is used, much of the information in the `aCreateInfo` structure
         * is overridden by the corresponding information in `aImageData`. The structure `aImageData` should be a
         * 2 dimensional image whose width and height should correspond to the highest level of detail.  The pixel data in
         * `aImageData` is copied to the main layer in the mipchain. The other mips are not generated automatically
         *
         * @param aCreateInfo Creation structure
         * @param aImageData  Image pixel data.
         */
        TextureData( sTextureCreateInfo const &aCreateInfo, sImageData const &aImageData );

        /** @brief Construct a texture from a create information structure and initial data from image file
         *
         * Note that when this version of the constructor is used, much of the information in the `aCreateInfo` structure
         * is overridden by the corresponding information in the image file `aImageData`. If `aImagePath` referes to a ktx,
         * kmg ot dds file, we use gli's `load` function to create the internal texture, and the loaded texture is considered
         * the authority on the texture's metadata, and the `aCreateInfo` data is overridden accordingly. If `aImagePath`
         * referes to a standard image file format, we use stb_image to load the data, which is assumed to be a two-dimensional
         * image (for now). Again in this case the size and format information is taken from the image file, and the mip chain
         * is not generated.
         *
         * @param aCreateInfo Creation structure
         * @param aImagePath  Image file.
         */
        TextureData( sTextureCreateInfo const &aCreateInfo, fs::path const &aImagePath );

        TextureData( char const *aKTXData, uint32_t aSize );

        void SaveTo( fs::path const &aImagePath );

        std::vector<char> Serialize() const;

      private:
        // Common initialization for all constructor versions.
        void Initialize();

      protected:
        // Internal texture structure.
        gli::texture mInternalTexture{};
    };

    /** \class TextureSampler2D
     *
     * @brief Sampler structure for CPU-side textures.
     *
     * This class abstracts away the information required to fetch samples from a texture. We note that
     * textures are sampled in a unit square or cube. For simplicity, since our textures are meant to
     * represent data in arbitrary rectangles, we add a scaling parameter which allow sampling parameters
     * in more general rectangles.
     */
    class TextureSampler2D;

    /** \class TextureData2D
     *
     * @brief Abstraction class for 2D textures.
     *
     * This specialization of the texture class is for 2 dimensional images, possibly with attached mip chains.
     * This class can be viewed as a layer of abstraction above gli's `texture2d` class.
     */
    class TextureData2D : public TextureData
    {
      public:
        friend class TextureSampler2D;

        /** @brief Default constructor */
        TextureData2D() = default;

        /** @brief Default destructor */
        ~TextureData2D() = default;

        /** @brief Other constructors
         *
         * These constructors perform the same function as the corresponding constructor for the base class, but also internally create
         * a 2D texture.
         */
        TextureData2D( sTextureCreateInfo const &aCreateInfo );
        TextureData2D( sTextureCreateInfo const &aCreateInfo, sImageData const &aImageData );
        TextureData2D( sTextureCreateInfo const &aCreateInfo, fs::path const &aImagePath );
        TextureData2D( char const *aKTXData, uint32_t aSize );

        sImageData GetImageData();

      protected:
        gli::texture2d mInternalTexture2d{};
    };

    /** \class TextureSampler2D
     *
     * @brief Abstraction class for 2D texture sampler.
     *
     * Modifies gli's `sampler2d` class to accept our version of the texture2d class. Sampling is the process
     * by which data is retrieved from a texture, with possible type conversions and interpolation.
     *
     */
    class TextureSampler2D : public gli::sampler2d<float>
    {
      public:
        sTextureSamplingInfo mSamplingSpec; //!< Sampling specification

        /** @brief Default constructor. */
        TextureSampler2D() = default;

        /** @brief Default destructor. */
        ~TextureSampler2D() = default;

        /** @brief Create a new sampler for the given texture and configuration
         *
         * @param aTexture Reference to the base texture object to sample
         * @param aSamplingInfo Information about how to sample the texture.
         *
         */
        TextureSampler2D( TextureData2D const &aTexture, sTextureSamplingInfo const &aSamplingInfo );
    };

    using sCubeMapImageData     = std::array<sImageData, 6>;
    using sCubeMapImagePathData = std::array<fs::path, 6>;

    class TextureDataCubeMap : public TextureData
    {
      public:
        friend class TextureSampler2D;

        /** @brief Default constructor */
        TextureDataCubeMap() = default;

        /** @brief Default destructor */
        ~TextureDataCubeMap() = default;

        TextureDataCubeMap( sTextureCreateInfo const &aCreateInfo );
        TextureDataCubeMap( sTextureCreateInfo const &aCreateInfo, sCubeMapImageData const &aImageData );
        TextureDataCubeMap( sTextureCreateInfo const &aCreateInfo, fs::path const &aKTXImagePath );
        TextureDataCubeMap( sTextureCreateInfo const &aCreateInfo, sCubeMapImagePathData const &aImagePaths );
        TextureDataCubeMap( std::vector<uint8_t> aKTXData, uint32_t aSize );
        TextureDataCubeMap( std::array<std::vector<uint8_t>, 6> aKTXData, uint32_t aSize );

        sCubeMapImageData GetImageData();

      protected:
        gli::texture_cube mInternalTextureCubeMap{};
    };

} // namespace SE::Core
