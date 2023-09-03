/// @file   TextureCubeMap.h
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
#include "Core/Memory.h"

#include "Core/Math/Types.h"

#include "Texture2D.h"

/** \namespace SE::Cuda
 */
namespace SE::Cuda
{

    using namespace SE::Core;

    class TextureSamplerCubeMap;

    /** \class TextureSamplerCubeMap
     *
     * @brief Sampler structure for CUDA textures.
     *
     * This class abstracts away the information required to fetch samples from a texture. We note that
     * textures are sampled in a unit square or cube. For simplicity, since our textures are meant to
     * represent data in arbitrary rectangles, we add a scaling parameter which allow sampling parameters
     * in more general rectangles.
     */
    class TextureSamplerCubeMap
    {
      public:
        sTextureSamplingInfo mSpec{};            //!< Copy of the specification structure used to create the texture
        ref_t<Texture2D>       mTexture = nullptr; //!< Reference to the parent texture

        struct DeviceData
        {
            TextureObject mTextureObject = 0; //!< Cuda-side sampler object

            math::vec2 mScaling = { 1.0f, 1.0f };
            math::vec2 mOffset  = { 0.0f, 0.0f };

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
        TextureSamplerCubeMap() = default;

        /** @brief Default destructor */
        ~TextureSamplerCubeMap() = default;

        /** @brief Create a new sampler for the given texture and configuration
         *
         * @param aTexture Texture to sample
         * @param aSamplingInfo Sampling data
         */
        TextureSamplerCubeMap( ref_t<Texture2D> &aTexture, const sTextureSamplingInfo &aSamplingInfo );

        void InitializeTextureSampler();
    };

    // cudaChannelFormatDesc ToCudaChannelDesc( eColorFormat aColorFormat );

} // namespace SE::Cuda
