#pragma once

// #define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <filesystem>

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Math/Types.h"
#include "ImporterData.h"

namespace fs = std::filesystem;

template <>
struct std::less<tinyobj::index_t>
{
    constexpr bool operator()( const tinyobj::index_t &a, const tinyobj::index_t &b ) const
    {
        if( a.vertex_index < b.vertex_index )
            return true;
        if( a.vertex_index > b.vertex_index )
            return false;

        if( a.normal_index < b.normal_index )
            return true;
        if( a.normal_index > b.normal_index )
            return false;

        if( a.texcoord_index < b.texcoord_index )
            return true;
        if( a.texcoord_index > b.texcoord_index )
            return false;

        return false;
    }
};

namespace SE::Core
{
    inline bool operator<( const tinyobj::index_t &a, const tinyobj::index_t &b );

    class ObjImporter : public sImportedModel
    {
      public:
        ObjImporter( const fs::path &aObjFile );
        ~ObjImporter() = default;

      private:
        // int AddVertex( const tinyobj::index_t &aIdx );
        int LoadTexture( const string_t &aInFileName, const string_t &aModelPath );

      private:
        fs::path mModelDir;

        tinyobj::attrib_t mAttributes;

        std::unordered_map<string_t, uint32_t> mKnownTextures;
        std::map<tinyobj::index_t, uint32_t>      mKnownVertices;
        std::vector<tinyobj::shape_t>             mShapes;

        std::unordered_map<string_t, uint32_t> mTextureLookup;
        std::unordered_map<int, uint32_t>         mMaterialIDLookup;

        std::vector<math::vec3> mVertexData;
        std::vector<math::vec3> mNormalsData;
        std::vector<math::vec2> mTexCoordData;

      private:
        void                                 AddVertex( sImportedMesh &aMesh, const tinyobj::index_t &aIdx );
        sImportedMaterial::sTextureReference RetrieveTextureData( string_t const &aTextureName );
        sImportedMaterial::sTextureReference PackMetalRoughTexture( string_t const &aMetalTextureName,
                                                                    string_t const &aRoughTextureName );
    };
} // namespace SE::Core
