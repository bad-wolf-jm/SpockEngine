#pragma once

#include <filesystem>
#include <string>

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGraphicsPipeline.h"
#include "Scene/MaterialSystem/MaterialSystem.h"
#include "yaml-cpp/yaml.h"

namespace fs = std::filesystem;

namespace SE::Core
{
    using namespace math;

    // Changing this value here also requires changing it in the vertex shader
    constexpr uint32_t MAX_NUM_JOINTS = 128u;

    struct sImportedTexture
    {
        std::string mName = "";

        Ref<TextureData2D>    mTexture = nullptr;
        Ref<TextureSampler2D> mSampler = nullptr;
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

        std::string mName = "";

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
        std::vector<float>     mInputs        = {};
        std::vector<glm::vec4> mOutputsVec4   = {};
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
        std::string                            mName     = "";
        std::vector<sImportedAnimationSampler> mSamplers = {};
        std::vector<sImportedAnimationChannel> mChannels = {};
        float                                  mStart    = std::numeric_limits<float>::max();
        float                                  mEnd      = std::numeric_limits<float>::min();
    };

    struct sImportedSkin
    {
        std::string           mName                = "";
        uint32_t              mSkeletonRootNodeID  = 0;
        std::vector<uint32_t> mJointNodeID         = {};
        std::vector<mat4>     mInverseBindMatrices = {};
    };

    struct sImportedMesh
    {
        std::string mName = "";

        Graphics::ePrimitiveTopology mPrimitive = Graphics::ePrimitiveTopology::TRIANGLES;

        uint32_t              mMaterialID = 0;
        std::vector<uint32_t> mIndices    = {};
        std::vector<vec3>     mPositions  = {};
        std::vector<vec3>     mNormals    = {};
        std::vector<vec2>     mUV0        = {};
        std::vector<vec2>     mUV1        = {};
        std::vector<uvec4>    mJoints     = {};
        std::vector<vec4>     mWeights    = {};
    };

    struct sImportedNode
    {
        std::string           mName      = "";
        uint32_t              mParentID  = std::numeric_limits<uint32_t>::max();
        uint32_t              mSkinID    = std::numeric_limits<uint32_t>::max();
        mat4                  mTransform = mat4( 1.0f );
        std::vector<uint32_t> mChildren  = {};
        std::vector<uint32_t> mMeshes    = {};
    };

    struct sImportedModel
    {
        std::vector<sImportedTexture>   mTextures   = {};
        std::vector<sImportedMaterial>  mMaterials  = {};
        std::vector<sImportedAnimation> mAnimations = {};
        std::vector<sImportedMesh>      mMeshes     = {};
        std::vector<sImportedNode>      mNodes      = {};
        std::vector<sImportedSkin>      mSkins      = {};
    };
} // namespace SE::Core