#pragma once

#include "Core/Math/Types.h"

#ifndef __CUDACC__
#    include "Graphics/API.h"
#endif

#include "Core/CUDA/Array/CudaBuffer.h"

using namespace math::literals;

#ifndef __CUDACC__
using namespace SE::Graphics;
using namespace math;
namespace fs = std::filesystem;
#endif

namespace SE::Core
{

    struct sShaderMaterial
    {
        vec4 mBaseColorFactor    = { 1.0f, 1.0f, 1.0f, 1.0f };
        int  mBaseColorTextureID = 0;
        int  mBaseColorUVChannel = 0;

        float mMetallicFactor     = 0.0f;
        float mRoughnessFactor    = 1.0f;
        int   mMetalnessUVChannel = 0;
        int   mMetalnessTextureID = 0;

        float mOcclusionStrength  = 0.0f;
        int   mOcclusionUVChannel = 0;
        int   mOcclusionTextureID = 0;

        alignas( 16 ) vec4 mEmissiveFactor = { 0.0f, 0.0f, 0.0f, 0.0f };
        int mEmissiveTextureID             = 0;
        int mEmissiveUVChannel             = 0;

        int mNormalTextureID = 0;
        int mNormalUVChannel = 0;

        float mAlphaThreshold = 0.0f;
    };

#ifndef __CUDACC__
    enum class eMaterialType : uint8_t
    {
        Opaque,
        Mask,
        Blend
    };

    struct sTextureReference
    {
        uint32_t mUVChannel = 0;
        uint32_t mTextureID = 0;
    };

    struct sMaterial
    {
        uint32_t    mID   = std::numeric_limits<uint32_t>::max();
        std::string mName = "";

        eMaterialType mType = eMaterialType::Opaque;

        float mLineWidth      = 1.0f;
        bool  mIsTwoSided     = true;
        bool  mUseAlphaMask   = false;
        float mAlphaThreshold = 0.5;

        vec4              mBaseColorFactor = 0xffffffff_rgbaf;
        sTextureReference mBaseColorTexture{};

        vec4              mEmissiveFactor = 0x00000000_rgbaf;
        sTextureReference mEmissiveTexture{};

        float             mRoughnessFactor = 1.0f;
        float             mMetallicFactor  = 1.0f;
        sTextureReference mMetalRoughTexture{};

        float             mOcclusionStrength = 1.0f;
        sTextureReference mOcclusionTexture{};

        sTextureReference mNormalsTexture{};

        sMaterial()                    = default;
        sMaterial( const sMaterial & ) = default;
    };
#endif
} // namespace SE::Core