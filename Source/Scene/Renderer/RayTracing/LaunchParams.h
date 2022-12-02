#pragma once

#include "Core/Math/Types.h"
#include "Core/Optix/Optix7.h"

#include "Core/Cuda/Texture2D.h"

#include "Scene/MaterialSystem/MaterialSystem.h"
#include "Scene/VertexData.h"

namespace SE::Core
{
    // for this simple example, we have a single ray type
    enum
    {
        RADIANCE_RAY_TYPE = 0,
        SHADOW_RAY_TYPE,
        RAY_TYPE_COUNT
    };

    struct sTriangleMeshSBTData
    {
        math::vec3   mColor;
        math::vec3  *mVertex;
        math::vec3  *mNormal;
        math::vec2  *mTexCoord;
        math::ivec3 *mIndex;
        bool         mHasTexture;

        cudaTextureObject_t mTexture;

        uint32_t mMaterialID;
        uint32_t mVertexOffset;
        uint32_t mIndexOffset;
    };

    struct sLaunchParams
    {
        int mNumPixelSamples = 8;
        int mNumLightSamples = 8;
        struct
        {
            int         mFrameID = 0;
            math::vec4 *mColorBuffer;
            math::vec4 *mNormalBuffer;
            math::vec4 *mAlbedoBuffer;

            math::ivec2 mSize;
            int         mAccumID{ 0 };
        } mFrame;

        struct
        {
            math::vec3 mPosition;
            math::vec3 mDirection;
            math::vec3 mHorizontal;
            math::vec3 mVertical;
        } mCamera;

        struct
        {
            math::vec3 mOrigin, mDu, mDv, mPower;
        } mLight;

        OptixTraversableHandle mSceneRoot;

        VertexData      *mVertexBuffer = nullptr;
        math::uvec3     *mIndexBuffer  = nullptr;
        sShaderMaterial *mMaterials    = nullptr;

        Cuda::TextureSampler2D::DeviceData *mTextures = nullptr;
    };

} // namespace SE::Core
