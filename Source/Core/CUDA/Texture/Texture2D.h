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
#include "Core/Definitions.h"
#include "Core/Logging.h"

#include "Conversion.h"
#include "Core/Math/Types.h"

/** \namespace SE::Cuda
 */
namespace SE::Cuda
{

    using namespace SE::Core;

    class texture_sampler2d_t;

    /** \class Texture2D
     *
     * @brief Abstraction class for Cuda textures.
     *
     * Textures are 2D or 3D arrays of numbers or vectors. Typically we think of textures as representing
     * images used for shading in graphics pipelines. In our case, we make use of the built-in interpolation
     * abilities of textures to represent functions @f$ f:[0, 1]^\ell\to \mathbb{R}^k @f$, where
     * @f$ \ell\in\{2, 3\} @f$ and  @f$ k\in\{1, 2, 3, 4\} @f$ which have been sampled on a discrete
     * rectangular lattice @f$ \Lambda \subseteq [0, 1]^\ell @f$.
     *
     * The texture along with all associated data is automatically destroyed when the object is deleted.
     */
    class texture2d_t
    {
        friend class texture_sampler2d_t;
        friend class texture_sampler_cubemap_t;

      public:
        texture_create_info_t mSpec; //!< Copy of the specification structure used to create the texture

        /** @brief Constructor
         *
         * Create a texture from the provided raw data, according to the requested specification
         *
         * @param aSpec Texture specification
         * @param aData Texture data
         */
        texture2d_t() = default;

        /** @brief Constructor
         *
         * Create a texture from the provided raw data, according to the requested specification
         *
         * @param aSpec Texture specification
         * @param aData Texture data
         */
        texture2d_t( texture_create_info_t &aSpec, vector_t<uint8_t> aData );

        /** @brief Constructor
         *
         * Create a texture from the provided raw data, according to the requested specification
         *
         * @param aSpec Texture specification
         * @param aData Texture data
         * @param aSize Data size, in bytes
         */
        texture2d_t( texture_create_info_t &aSpec, uint8_t *aData, size_t aSize );

        /** @brief Constructor
         *
         * Create a texture from the provided imagedata, according to the requested specification.
         *
         * @param aSpec      Texture specification
         * @param aImageData Texture image data
         */
        texture2d_t( texture_create_info_t &aSpec, image_data_t &aImageData );

        /** @brief Constructor
         *
         * Create a texture from the provided imagedata, according to the requested specification.
         *
         * @param aSpec      Texture specification
         * @param aImageData Texture image data
         */
        texture2d_t( texture_create_info_t &aSpec, void *aExternalBuffer, size_t aImageMemorySize );

        /** @brief Constructor
         *
         * Create a texture from the provided imagedata, according to the requested specification.
         *
         * @param aSpec      Texture specification
         * @param aImageData Texture image data
         */
        // Texture2D( sTextureCreateInfo &aSpec, Graphics::Texture2D &aImageData );

        /** @brief Destructor */
        ~texture2d_t();

      protected:
        size_t            mImageMemorySize            = 0;
        array_t           mInternalCudaArray          = nullptr;
        mipmapped_array_t mInternalCudaMipmappedArray = nullptr;
        external_memory_t mExternalMemoryHandle       = nullptr;
    };

    /** \class texture_data_sampler2d_t
     *
     * @brief Sampler structure for CUDA textures.
     *
     * This class abstracts away the information required to fetch samples from a texture. We note that
     * textures are sampled in a unit square or cube. For simplicity, since our textures are meant to
     * represent data in arbitrary rectangles, we add a scaling parameter which allow sampling parameters
     * in more general rectangles.
     */
    class texture_sampler2d_t
    {
      public:
        texture_sampling_info_t mSpec{};            //!< Copy of the specification structure used to create the texture
        ref_t<texture2d_t>      mTexture = nullptr; //!< Reference to the parent texture

        struct DeviceData
        {
            texture_object_t mTextureObject = 0; //!< Cuda-side sampler object
            math::vec2       mScaling       = { 1.0f, 1.0f };
            math::vec2       mOffset        = { 0.0f, 0.0f };

            /** @brief Retrieve en element from the texture
             *
             * @param x x coordinate of the texel to retrieve.
             * @param y y coordinate of the texel to retrieve
             */
            template <typename _Ty>
            SE_CUDA_DEVICE_FUNCTION_DEF _Ty Fetch( float x, float y )
            {
                return tex2D<_Ty>( mTextureObject, ( x + mOffset.x ) / mScaling.x, ( y + mOffset.y ) / mScaling.y );
            }

        } mDeviceData;

        /** @brief Default constructor */
        texture_sampler2d_t() = default;

        /** @brief Default destructor */
        ~texture_sampler2d_t() = default;

        /** @brief Create a new sampler for the given texture and configuration
         *
         * @param aTexture Texture to sample
         * @param aSamplingInfo Sampling data
         */
        texture_sampler2d_t( ref_t<texture2d_t> &aTexture, const texture_sampling_info_t &aSamplingInfo );
        void InitializeTextureSampler();
    };
} // namespace SE::Cuda
