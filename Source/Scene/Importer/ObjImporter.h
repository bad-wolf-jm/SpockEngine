#pragma once

// #define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Textures/TextureData.h"
#include "Core/Textures/TextureTypes.h"
#include "ImporterData.h"

namespace fs = std::filesystem;

namespace SE::Core
{

    class ObjImporter : public sImportedModel
    {
      public:
        ObjImporter( const fs::path &aObjFile );
        ~ObjImporter() = default;

      private:
        int AddVertex( const tinyobj::index_t &aIdx );
        int LoadTexture( const std::string &aInFileName, const std::string &aModelPath );

      private:
        fs::path mModelDir;

        std::unordered_map<std::string, uint32_t> mKnownTextures;
        std::map<tinyobj::index_t, uint32_t>      mKnownVertices;
        tinyobj::attrib_t                         mAttributes;
        std::vector<tinyobj::shape_t>             mShapes;

        std::unordered_map<std::string, uint32_t> mTextureLookup;

        std::vector<math::vec3> mVertexData;
        std::vector<math::vec3> mNormalsData;
        std::vector<math::vec2> mTexCoordData;

      private:
        sImportedMaterial::sTextureReference RetrieveTextureData( std::string const &aTextureName );
        sImportedMaterial::sTextureReference PackMetalRoughTexture( std::string const &aMetalTextureName,
                                                                    std::string const &aRoughTextureName );
    };
} // namespace SE::Core
