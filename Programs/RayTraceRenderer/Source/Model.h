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

#include "Core/Math/Types.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
    using namespace gdt;

    /*! a simple indexed triangle mesh that our sample renderer will
        render */
    struct TriangleMesh
    {
        std::vector<math::vec3>  mVertex;
        std::vector<math::vec3>  mNormal;
        std::vector<math::vec2>  mTexCoord;
        std::vector<math::ivec3> mIndex;

        // material data:
        math::vec3 mDiffuse;
        int        mDiffuseTextureID{ -1 };
    };

    struct QuadLight
    {
        math::vec3 mOrigin, mDu, mDv, mPower;
    };

    struct Texture
    {
        ~Texture()
        {
            if( mPixel ) delete[] mPixel;
        }

        uint32_t   *mPixel{ nullptr };
        math::ivec2 mResolution{ -1 };
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
        box_t<math::vec3> mBounds;
    };

    Model *loadOBJ( const std::string &objFile );
} // namespace osc
