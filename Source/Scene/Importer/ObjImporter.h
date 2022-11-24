#pragma once
#include "tiny_gltf.h"
#include <filesystem>

#include "Core/Textures/TextureTypes.h"
#include "ImporterData.h"

namespace fs = std::filesystem;

namespace SE::Core
{

    class ObjImporter : public sImportedModel
    {
      public:
        ObjImporter( fs::path aObjFile );
        ~ObjImporter() = default;

      private:
        int AddVertex( TriangleMesh *aMesh, tinyobj::attrib_t &aAttributes, const tinyobj::index_t &aIdx, );
        int LoadTexture( const std::string &aInFileName, const std::string &aModelPath );

      private:
        std::map<std::string, int>       mKnownTextures;
        std::map<tinyobj::index_t, int>  mKnownVertices;
        tinyobj::attrib_t                mAttributes;
        std::vector<tinyobj::shape_t>    mShapes;
        std::vector<tinyobj::material_t> mMaterials;
        fs::path                         mModelDir;

        std::unordered_map<std::string, uint32_t> mTextureLookup;

        std:::vector<math::vec3> mVertexData;
        std:::vector<math::vec3> mNormalsData;
        std:::vector<math::vec2> mTexCoordData;
    };
} // namespace SE::Core