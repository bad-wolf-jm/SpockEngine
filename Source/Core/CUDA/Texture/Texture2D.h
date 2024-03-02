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

#include "Conversion.h"
#include "Core/Math/Types.h"

/** \namespace SE::Cuda
 */
namespace SE::Cuda
{

    using namespace SE::Core;

    class TextureSampler2D;

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
    class Texture2D
    {
        friend class TextureSampler2D;
        friend class TextureSamplerCubeMap;

      public:
        sTextureCreateInfo mSpec; //!< Copy of the specification structure used to create the texture

        /** @brief Constructor
         *
         * Create a texture from the provided raw data, according to the requested specification
         *
         * @param aSpec Texture specification
         * @param aData Texture data
         */
        Texture2D() = default;

        /** @brief Constructor
         *
         * Create a texture from the provided raw data, according to the requested specification
         *
         * @param aSpec Texture specification
         * @param aData Texture data
         */
        Texture2D( sTextureCreateInfo &aSpec, vector_t<uint8_t> aData );

        /** @brief Constructor
         *
         * Create a texture from the provided raw data, according to the requested specification
         *
         * @param aSpec Texture specification
         * @param aData Texture data
         * @param aSize Data size, in bytes
         */
        Texture2D( sTextureCreateInfo &aSpec, uint8_t *aData, size_t aSize );

        /** @brief Constructor
         *
         * Create a texture from the provided imagedata, according to the requested specification.
         *
         * @param aSpec      Texture specification
         * @param aImageData Texture image data
         */
        Texture2D( sTextureCreateInfo &aSpec, sImageData &aImageData );

        /** @brief Constructor
         *
         * Create a texture from the provided imagedata, according to the requested specification.
         *
         * @param aSpec      Texture specification
         * @param aImageData Texture image data
         */
        Texture2D( sTextureCreateInfo &aSpec, void *aExternalBuffer, size_t aImageMemorySize );

        /** @brief Constructor
         *
         * Create a texture from the provided imagedata, according to the requested specification.
         *
         * @param aSpec      Texture specification
         * @param aImageData Texture image data
         */
        // Texture2D( sTextureCreateInfo &aSpec, Graphics::Texture2D &aImageData );

        /** @brief Destructor */
        ~Texture2D();

      protected:
        size_t         mImageMemorySize            = 0;
        Array          mInternalCudaArray          = nullptr;
        MipmappedArray mInternalCudaMipmappedArray = nullptr;
        ExternalMemory mExternalMemoryHandle       = nullptr;
    };

    /** \class TextureSampler2D
     *
     * @brief Sampler structure for CUDA textures.
     *
     * This class abstracts away the information required to fetch samples from a texture. We note that
     * textures are sampled in a unit square or cube. For simplicity, since our textures are meant to
     * represent data in arbitrary rectangles, we add a scaling parameter which allow sampling parameters
     * in more general rectangles.
     */
    class TextureSampler2D
    {
      public:
        sTextureSamplingInfo mSpec{};            //!< Copy of the specification structure used to create the texture
        ref_t<Texture2D>       mTexture = nullptr; //!< Reference to the parent texture

        struct DeviceData
        {
            TextureObject mTextureObject = 0; //!< Cuda-side sampler object
            math::vec2    mScaling       = { 1.0f, 1.0f };
            math::vec2    mOffset        = { 0.0f, 0.0f };

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
        TextureSampler2D() = default;

        /** @brief Default destructor */
        ~TextureSampler2D() = default;

        /** @brief Create a new sampler for the given texture and configuration
         *
         * @param aTexture Texture to sample
         * @param aSamplingInfo Sampling data
         */
        TextureSampler2D( ref_t<Texture2D> &aTexture, const sTextureSamplingInfo &aSamplingInfo );
        void InitializeTextureSampler();
    };
} // namespace SE::Cuda
