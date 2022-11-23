// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "Core/Math/Types.h"
#include "Core/Optix/Optix7.h"

// #include "gdt/math/vec.h"

namespace osc
{
    // using namespace gdt;

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

            /*! the size of the frame buffer to render */
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
    };

} // namespace osc
