#pragma once
#include "tiny_gltf.h"
#include <filesystem>

#include "Core/CUDA/Texture/TextureTypes.h"
#include "ImporterData.h"

namespace fs = std::filesystem;

namespace SE::Core
{

    class GlTFImporter : public sImportedModel
    {
      public:
        GlTFImporter( fs::path aPath );
        ~GlTFImporter() = default;

      private:
        tinygltf::Model                        mGltfModel;
        vec_t<sTextureSamplingInfo>      mTextureSamplers  = {};
        std::unordered_map<uint32_t, uint32_t> mTextureIDLookup  = {};
        std::unordered_map<uint32_t, uint32_t> mMaterialIDLookup = {};
        std::unordered_map<uint32_t, uint32_t> mNodeIDLookup     = {};
        std::unordered_map<uint32_t, uint32_t> mSamplerIDLookup  = {};
        std::unordered_map<uint32_t, uint32_t> mSkinIDLookup     = {};

      private:
        void LoadSamplers();
        void CreateTexture( uint32_t aTextureIndex, string_t aName, tinygltf::Image const &aGltfimage,
                            sTextureSamplingInfo const &aTextureSamplingInfo );
        void LoadTextures();
        sImportedMaterial::sTextureReference RetrieveTextureData( tinygltf::Material &aMaterial, string_t aName );
        sImportedMaterial::sTextureReference RetrieveAdditionalTextureData( tinygltf::Material &aMaterial, string_t aName );
        template <typename _Ty>
        _Ty RetrieveValue( tinygltf::Material &aMaterial, string_t aName, _Ty aDefault )
        {
            if( aMaterial.values.find( aName ) != aMaterial.values.end() )
            {
                return static_cast<_Ty>( aMaterial.values[aName].Factor() );
            }

            return aDefault;
        }
        math::vec4 RetrieveVec4( tinygltf::Material &aMaterial, string_t aName, math::vec4 aDefault );

        template <typename _Ty>
        void RetrievePrimitiveAttribute( const tinygltf::Primitive &aPrimitive, string_t aName, vec_t<_Ty> &aOutput )
        {
            if( aPrimitive.attributes.find( aName ) != aPrimitive.attributes.end() )
            {
                const tinygltf::Accessor   &lAccessor   = mGltfModel.accessors[aPrimitive.attributes.find( aName )->second];
                const tinygltf::BufferView &lBufferView = mGltfModel.bufferViews[lAccessor.bufferView];

                auto *lBufferData = &( mGltfModel.buffers[lBufferView.buffer].data[lAccessor.byteOffset + lBufferView.byteOffset] );

                auto lBufferCount      = static_cast<uint32_t>( lAccessor.count );
                auto lBufferByteStride = lAccessor.ByteStride( lBufferView );

                aOutput.resize( lBufferCount * lBufferByteStride / sizeof( _Ty ) );
                memcpy( aOutput.data(), lBufferData, lBufferCount * lBufferByteStride );
            }
        }

        std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> RetrievePrimitiveCount( const tinygltf::Primitive &aPrimitive,
                                                                                   string_t                aName );

        void LoadMaterials();

        void LoadNode( uint32_t aParentID, tinygltf::Node const &aNode, uint32_t aNodeID );
        void LoadNodes();

        void LoadAnimations();
        void LoadSkins();
    };
} // namespace SE::Core