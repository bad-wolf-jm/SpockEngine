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
#include "Core/Definitions.h"
#include "Core/Logging.h"

#include "Core/Math/Types.h"

#include "Texture2D.h"

/** \namespace SE::Cuda
 */
namespace SE::Cuda
{

    using namespace SE::Core;

    class texture_sampler_cubemap_t;

    /** \class TextureSamplerCubeMap
     *
     * @brief Sampler structure for CUDA textures.
     *
     * This class abstracts away the information required to fetch samples from a texture. We note that
     * textures are sampled in a unit square or cube. For simplicity, since our textures are meant to
     * represent data in arbitrary rectangles, we add a scaling parameter which allow sampling parameters
     * in more general rectangles.
     */
    class texture_sampler_cubemap_t
    {
      public:
        texture_sampling_info_t mSpec{};            //!< Copy of the specification structure used to create the texture
        ref_t<texture2d_t>      mTexture = nullptr; //!< Reference to the parent texture

        struct DeviceData
        {
            texture_object_t mTextureObject = 0; //!< Cuda-side sampler object

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
        texture_sampler_cubemap_t() = default;

        /** @brief Default destructor */
        ~texture_sampler_cubemap_t() = default;

        /** @brief Create a new sampler for the given texture and configuration
         *
         * @param aTexture Texture to sample
         * @param aSamplingInfo Sampling data
         */
        texture_sampler_cubemap_t( ref_t<texture2d_t> &aTexture, const texture_sampling_info_t &aSamplingInfo );

        void InitializeTextureSampler();
    };

    // cudaChannelFormatDesc ToCudaChannelDesc( eColorFormat aColorFormat );

} // namespace SE::Cuda
