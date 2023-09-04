#pragma once

#include <filesystem>
#include <string>

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Graphics/API.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Scene/MaterialSystem/MaterialSystem.h"
#include "yaml-cpp/yaml.h"

namespace fs = std::filesystem;

namespace SE::Core
{
    using namespace math;

    // Changing this value here also requires changing it in the vertex shader
    constexpr uint32_t MAX_NUM_JOINTS = 128u;

    struct sImportedTexture
    {
        string_t mName = "";

        ref_t<TextureData2D>    mTexture = nullptr;
        ref_t<TextureSampler2D> mSampler = nullptr;
    };

    struct sImportedMaterial
    {
        enum class AlphaMode : uint8_t
        {
            OPAQUE_MODE     = 0,
            ALPHA_MASK_MODE = 1,
            BLEND_MODE      = 2
        };

        enum class PBRWorkflow : uint8_t
        {
            METAL_ROUGH    = 0,
            SPECULAR_GLOSS = 1
        };

        struct sTextureReference
        {
            uint32_t TextureID = std::numeric_limits<uint32_t>::max();
            uint32_t UVChannel = 0;
        };

        string_t mName = "";

        struct
        {
            AlphaMode mMode   = AlphaMode::OPAQUE_MODE;
            float     mCutOff = 1.0f;
        } mAlpha;

        struct
        {
            bool  mIsTwoSided      = false;
            float mMetallicFactor  = 1.0f;
            float mRoughnessFactor = 1.0f;
            vec4  mBaseColorFactor = vec4( 1.0f );
            vec4  mEmissiveFactor  = vec4( 1.0f );
        } mConstants;

        struct
        {
            sTextureReference mBaseColorTexture{};
            sTextureReference mMetallicRoughnessTexture{};
            sTextureReference mNormalTexture{};
            sTextureReference mOcclusionTexture{};
            sTextureReference mEmissiveTexture{};
        } mTextures;
    };

    struct sImportedAnimationSampler
    {
        enum class Interpolation : uint8_t
        {
            LINEAR,
            STEP,
            CUBICSPLINE
        };

        Interpolation          mInterpolation = Interpolation::LINEAR;
        vector_t<float>     mInputs        = {};
        vector_t<glm::vec4> mOutputsVec4   = {};
    };

    struct sImportedAnimationChannel
    {
        enum class Channel : uint8_t
        {
            TRANSLATION,
            ROTATION,
            SCALE
        };

        Channel  mComponent    = Channel::TRANSLATION;
        uint32_t mNodeID       = std::numeric_limits<uint32_t>::max();
        uint32_t mSamplerIndex = std::numeric_limits<uint32_t>::max();
    };

    struct sImportedAnimation
    {
        string_t                            mName     = "";
        vector_t<sImportedAnimationSampler> mSamplers = {};
        vector_t<sImportedAnimationChannel> mChannels = {};
        float                                  mStart    = std::numeric_limits<float>::max();
        float                                  mEnd      = std::numeric_limits<float>::min();
    };

    struct sImportedSkin
    {
        string_t           mName                = "";
        uint32_t              mSkeletonRootNodeID  = 0;
        vector_t<uint32_t> mJointNodeID         = {};
        vector_t<mat4>     mInverseBindMatrices = {};
    };

    struct sImportedMesh
    {
        string_t mName = "";

        Graphics::ePrimitiveTopology mPrimitive = Graphics::ePrimitiveTopology::TRIANGLES;

        uint32_t              mMaterialID = 0;
        vector_t<uint32_t> mIndices    = {};
        vector_t<vec3>     mPositions  = {};
        vector_t<vec3>     mNormals    = {};
        vector_t<vec2>     mUV0        = {};
        vector_t<vec2>     mUV1        = {};
        vector_t<uvec4>    mJoints     = {};
        vector_t<vec4>     mWeights    = {};
    };

    struct sImportedNode
    {
        string_t           mName      = "";
        uint32_t              mParentID  = std::numeric_limits<uint32_t>::max();
        uint32_t              mSkinID    = std::numeric_limits<uint32_t>::max();
        mat4                  mTransform = mat4( 1.0f );
        vector_t<uint32_t> mChildren  = {};
        vector_t<uint32_t> mMeshes    = {};
    };

    struct sImportedModel
    {
        vector_t<sImportedTexture>   mTextures   = {};
        vector_t<sImportedMaterial>  mMaterials  = {};
        vector_t<sImportedAnimation> mAnimations = {};
        vector_t<sImportedMesh>      mMeshes     = {};
        vector_t<sImportedNode>      mNodes      = {};
        vector_t<sImportedSkin>      mSkins      = {};
    };
} // namespace SE::Core