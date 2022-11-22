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

#include "gdt/math/AffineSpace.h"
#include <vector>

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
    using namespace gdt;

    /*! a simple indexed triangle mesh that our sample renderer will
        render */
    struct TriangleMesh
    {
        std::vector<vec3f> mVertex;
        std::vector<vec3f> mNormal;
        std::vector<vec2f> mTexCoord;
        std::vector<vec3i> mIndex;

        // material data:
        vec3f mDiffuse;
        int   mDiffuseTextureID{ -1 };
    };

    struct QuadLight
    {
        vec3f mOrigin, mDu, mDv, mPower;
    };

    struct Texture
    {
        ~Texture()
        {
            if( mPixel ) delete[] mPixel;
        }

        uint32_t *mPixel{ nullptr };
        vec2i     mResolution{ -1 };
    };

    struct Model
    {
        ~Model()
        {
            for( auto mesh : mMeshes ) delete mesh;
            for( auto texture : mTextures ) delete texture;
        }

        std::vector<TriangleMesh *> mMeshes;
        std::vector<Texture *>      mTextures;
        //! bounding box of all vertices in the model
        box3f mBounds;
    };

    Model *loadOBJ( const std::string &objFile );
} // namespace osc
