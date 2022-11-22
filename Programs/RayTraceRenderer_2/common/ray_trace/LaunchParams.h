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

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc
{
    using namespace gdt;

    // for this simple example, we have a single ray type
    enum
    {
        RADIANCE_RAY_TYPE = 0,
        SHADOW_RAY_TYPE,
        RAY_TYPE_COUNT
    };

    struct sTriangleMeshSBTData
    {
        vec3f               mColor;
        vec3f              *mVertex;
        vec3f              *mNormal;
        vec2f              *mTexCoord;
        vec3i              *mIndex;
        bool                mHasTexture;
        cudaTextureObject_t mTexture;
    };

    struct sLaunchParams
    {
        int mNumPixelSamples = 8;
        struct
        {
            int       mFrameID = 0;
            float4   *mColorBuffer;
            uint32_t *mColorBufferU32;
            float4   *mNormalBuffer;
            float4   *mAlbedoBuffer;

            /*! the size of the frame buffer to render */
            vec2i mSize;
            int   mAccumID{ 0 };
        } mFrame;

        struct
        {
            vec3f mPosition;
            vec3f mDirection;
            vec3f mHorizontal;
            vec3f mVertical;
        } mCamera;

        struct
        {
            vec3f mOrigin, mDu, mDv, mPower;
        } mLight;

        OptixTraversableHandle mSceneRoot;
    };

} // namespace osc
