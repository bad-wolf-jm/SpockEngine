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
