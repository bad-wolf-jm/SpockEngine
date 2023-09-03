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
        vec_t<float>     mInputs        = {};
        vec_t<glm::vec4> mOutputsVec4   = {};
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
        vec_t<sImportedAnimationSampler> mSamplers = {};
        vec_t<sImportedAnimationChannel> mChannels = {};
        float                                  mStart    = std::numeric_limits<float>::max();
        float                                  mEnd      = std::numeric_limits<float>::min();
    };

    struct sImportedSkin
    {
        string_t           mName                = "";
        uint32_t              mSkeletonRootNodeID  = 0;
        vec_t<uint32_t> mJointNodeID         = {};
        vec_t<mat4>     mInverseBindMatrices = {};
    };

    struct sImportedMesh
    {
        string_t mName = "";

        Graphics::ePrimitiveTopology mPrimitive = Graphics::ePrimitiveTopology::TRIANGLES;

        uint32_t              mMaterialID = 0;
        vec_t<uint32_t> mIndices    = {};
        vec_t<vec3>     mPositions  = {};
        vec_t<vec3>     mNormals    = {};
        vec_t<vec2>     mUV0        = {};
        vec_t<vec2>     mUV1        = {};
        vec_t<uvec4>    mJoints     = {};
        vec_t<vec4>     mWeights    = {};
    };

    struct sImportedNode
    {
        string_t           mName      = "";
        uint32_t              mParentID  = std::numeric_limits<uint32_t>::max();
        uint32_t              mSkinID    = std::numeric_limits<uint32_t>::max();
        mat4                  mTransform = mat4( 1.0f );
        vec_t<uint32_t> mChildren  = {};
        vec_t<uint32_t> mMeshes    = {};
    };

    struct sImportedModel
    {
        vec_t<sImportedTexture>   mTextures   = {};
        vec_t<sImportedMaterial>  mMaterials  = {};
        vec_t<sImportedAnimation> mAnimations = {};
        vec_t<sImportedMesh>      mMeshes     = {};
        vec_t<sImportedNode>      mNodes      = {};
        vec_t<sImportedSkin>      mSkins      = {};
    };
} // namespace SE::Core