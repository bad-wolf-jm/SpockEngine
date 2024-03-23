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

#include "Core/Definitions.h"

#include "ColorFormat.h"
#include "TextureTypes.h"

namespace fs = std::filesystem;

namespace SE::Core
{
#if 0
    /** \struct TextureData
     *
     * @brief CPU-side class for textures
     *
     */
    class texture_data_t
    {
      public:
        /** @brief Texture creation metadata */

      public:
        texture_create_info_t mSpec; /**!< Copy of the sCreateInfo structure used ot define the texture. */

        /** @brief Default constructor*/
        texture_data_t() = default;

        /** @brief Default destructor */
        ~texture_data_t() = default;

        /** @brief Construct a texture from a create information structure
         *
         * This also allocates the memory for the texture data, the amount of which is determined from the information
         * in the `aCreateInfo` structure. No initialization is performed by the constructor.
         *
         * @param aCreateInfo Creation structure
         */
        texture_data_t( texture_create_info_t const &aCreateInfo );

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
        texture_data_t( texture_create_info_t const &aCreateInfo, image_data_t const &aImageData );

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
        texture_data_t( texture_create_info_t const &aCreateInfo, fs::path const &aImagePath );

        texture_data_t( char const *aKTXData, uint32_t aSize );

        void SaveTo( fs::path const &aImagePath );

        vector_t<char> Serialize() const;

      private:
        // Common initialization for all constructor versions.
        void Initialize();

      protected:
        // Internal texture structure.
        gli::texture mInternalTexture{};
    };

    /** \class texture_data_sampler2d_t
     *
     * @brief Sampler structure for CPU-side textures.
     *
     * This class abstracts away the information required to fetch samples from a texture. We note that
     * textures are sampled in a unit square or cube. For simplicity, since our textures are meant to
     * represent data in arbitrary rectangles, we add a scaling parameter which allow sampling parameters
     * in more general rectangles.
     */
    class texture_data_sampler2d_t;

    /** \class TextureData2D
     *
     * @brief Abstraction class for 2D textures.
     *
     * This specialization of the texture class is for 2 dimensional images, possibly with attached mip chains.
     * This class can be viewed as a layer of abstraction above gli's `texture2d` class.
     */
    class texture_data2d_t : public texture_data_t
    {
      public:
        friend class texture_data_sampler2d_t;

        /** @brief Default constructor */
        texture_data2d_t() = default;

        /** @brief Default destructor */
        ~texture_data2d_t() = default;

        /** @brief Other constructors
         *
         * These constructors perform the same function as the corresponding constructor for the base class, but also internally create
         * a 2D texture.
         */
        texture_data2d_t( texture_create_info_t const &aCreateInfo );
        texture_data2d_t( texture_create_info_t const &aCreateInfo, image_data_t const &aImageData );
        texture_data2d_t( texture_create_info_t const &aCreateInfo, fs::path const &aImagePath );
        texture_data2d_t( char const *aKTXData, uint32_t aSize );

        image_data_t GetImageData();

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
    class texture_data_sampler2d_t : public gli::sampler2d<float>
    {
      public:
        texture_sampling_info_t mSamplingSpec; //!< Sampling specification

        /** @brief Default constructor. */
        texture_data_sampler2d_t() = default;

        /** @brief Default destructor. */
        ~texture_data_sampler2d_t() = default;

        /** @brief Create a new sampler for the given texture and configuration
         *
         * @param aTexture Reference to the base texture object to sample
         * @param aSamplingInfo Information about how to sample the texture.
         *
         */
        texture_data_sampler2d_t( texture_data2d_t const &aTexture, texture_sampling_info_t const &aSamplingInfo );
    };
#endif
    using cubemap_image_data_t      = std::array<image_data_t, 6>;
    using cubemap_image_path_data_t = std::array<fs::path, 6>;

#if 0
    class texture_data_cubemap_t : public texture_data_t
    {
      public:
        friend class texture_data_sampler2d_t;

        /** @brief Default constructor */
        texture_data_cubemap_t() = default;

        /** @brief Default destructor */
        ~texture_data_cubemap_t() = default;

        texture_data_cubemap_t( texture_create_info_t const &aCreateInfo );
        texture_data_cubemap_t( texture_create_info_t const &aCreateInfo, cubemap_image_data_t const &aImageData );
        texture_data_cubemap_t( texture_create_info_t const &aCreateInfo, fs::path const &aKTXImagePath );
        texture_data_cubemap_t( texture_create_info_t const &aCreateInfo, cubemap_image_path_data_t const &aImagePaths );
        texture_data_cubemap_t( vector_t<uint8_t> aKTXData, uint32_t aSize );
        texture_data_cubemap_t( std::array<vector_t<uint8_t>, 6> aKTXData, uint32_t aSize );

        cubemap_image_data_t GetImageData();

      protected:
        gli::texture_cube mInternalTextureCubeMap{};
    };
#endif
} // namespace SE::Core
