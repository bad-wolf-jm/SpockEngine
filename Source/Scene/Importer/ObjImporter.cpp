#include "ObjImporter.h"
#include <iostream>

#include "Core/Logging.h"
#include "fmt/core.h"

namespace SE::Core
{
    void ObjImporter::AddVertex( sImportedMesh &aMesh, const tinyobj::index_t &aIdx )
    {
        if( mKnownVertices.find( aIdx ) != mKnownVertices.end() )
        {
            aMesh.mIndices.push_back( mKnownVertices[aIdx] );
            return;
        }

        int lNewID           = (int)aMesh.mPositions.size();
        mKnownVertices[aIdx] = lNewID;

        aMesh.mPositions.push_back( mVertexData[aIdx.vertex_index] );
        if( aIdx.normal_index >= 0 )
            while( aMesh.mNormals.size() < aMesh.mPositions.size() )
                aMesh.mNormals.push_back( mNormalsData[aIdx.normal_index] );

        if( aIdx.texcoord_index >= 0 )
        {
            while( aMesh.mUV0.size() < aMesh.mPositions.size() )
            {
                aMesh.mUV0.push_back( mTexCoordData[aIdx.texcoord_index] );
            }
        }

        aMesh.mUV0.resize( aMesh.mPositions.size() );
        // aMesh.mUV1.resize( aMesh.mPositions.size() );
        aMesh.mJoints.resize( aMesh.mPositions.size() );
        aMesh.mWeights.resize( aMesh.mPositions.size() );
        aMesh.mNormals.resize( aMesh.mPositions.size() );

        aMesh.mIndices.push_back( mKnownVertices[aIdx] );
    }

    sImportedMaterial::sTextureReference ObjImporter::PackMetalRoughTexture( string_t const &aMetalTextureName,
                                                                             string_t const &aRoughTextureName )
    {
        return sImportedMaterial::sTextureReference{};
    }

    sImportedMaterial::sTextureReference ObjImporter::RetrieveTextureData( string_t const &aTextureName )
    {
        if( aTextureName.empty() )
            return sImportedMaterial::sTextureReference{};

        SE::Logging::Info( "Loading texture: {}", aTextureName );

        if( mKnownTextures.find( aTextureName ) == mKnownTextures.end() )
        {
            auto const lTexturePath = mModelDir / aTextureName;

            sImportedTexture lNewTexture{};
            lNewTexture.mName = aTextureName.empty() ? fmt::format( "TEXTURE_{}", mTextures.size() ) : aTextureName;

            lNewTexture.mTexture = New<TextureData2D>( sTextureCreateInfo{}, lTexturePath );

            sTextureSamplingInfo lSamplerCreateInfo{};
            lSamplerCreateInfo.mWrapping              = eSamplerWrapping::REPEAT;
            lSamplerCreateInfo.mNormalizedCoordinates = true;
            lSamplerCreateInfo.mNormalizedValues      = true;
            lNewTexture.mSampler                      = New<TextureSampler2D>( *lNewTexture.mTexture, lSamplerCreateInfo );

            mTextures.push_back( lNewTexture );
            mKnownTextures[aTextureName] = mTextures.size() - 1;
        }

        return sImportedMaterial::sTextureReference{ mKnownTextures[aTextureName], 0 };
    }

    ObjImporter::ObjImporter( const fs::path &aObjFile )
    {
        vec_t<tinyobj::shape_t>    lObjShapes;
        vec_t<tinyobj::material_t> lObjMaterials;

        mModelDir                = aObjFile.parent_path();
        string_t lErrorString = "";
        bool        lReadOK      = tinyobj::LoadObj( &mAttributes, &lObjShapes, &lObjMaterials, &lErrorString, &lErrorString,
                                                     aObjFile.string().c_str(), mModelDir.string().c_str(), true );
        if( !lReadOK )
            throw std::runtime_error( "Could not read OBJ model from " + aObjFile.string() + " : " + lErrorString );
        // if( lObjMaterials.empty() ) throw std::runtime_error( "Could not parse materials ..." );

        for( uint32_t i = 0; i < mAttributes.vertices.size() / 3; i++ )
            mVertexData.push_back(
                math::vec3{ mAttributes.vertices[3 * i + 0], mAttributes.vertices[3 * i + 1], mAttributes.vertices[3 * i + 2] } );

        for( uint32_t i = 0; i < mAttributes.normals.size() / 3; i++ )
            mNormalsData.push_back(
                math::vec3{ mAttributes.normals[3 * i + 0], mAttributes.normals[3 * i + 1], mAttributes.normals[3 * i + 2] } );

        for( uint32_t i = 0; i < mAttributes.texcoords.size() / 2; i++ )
            mTexCoordData.push_back( math::vec2{ mAttributes.texcoords[2 * i + 0], mAttributes.texcoords[2 * i + 1] } );

        std::cout << "Done loading obj file - found " << lObjShapes.size() << " shapes with " << lObjMaterials.size() << " materials"
                  << std::endl;

        if( lObjMaterials.empty() )
            return;

        uint32_t lMaterialIndex = 0;
        for( auto const &lMaterial : lObjMaterials )
        {
            sImportedMaterial lNewImportedMaterial{};

            lNewImportedMaterial.mName =
                lMaterial.name.empty() ? fmt::format( "UNNAMED_MATERIAL_{}", lMaterialIndex ) : lMaterial.name;

            lNewImportedMaterial.mConstants.mIsTwoSided      = false;
            lNewImportedMaterial.mConstants.mMetallicFactor  = lMaterial.metallic;
            lNewImportedMaterial.mConstants.mRoughnessFactor = lMaterial.roughness;

            lNewImportedMaterial.mConstants.mBaseColorFactor =
                math::vec4{ lMaterial.diffuse[0], lMaterial.diffuse[1], lMaterial.diffuse[2], 1.0f };
            lNewImportedMaterial.mConstants.mEmissiveFactor =
                math::vec4{ lMaterial.emission[0], lMaterial.emission[1], lMaterial.emission[2], 1.0f };

            lNewImportedMaterial.mTextures.mBaseColorTexture = RetrieveTextureData( lMaterial.diffuse_texname );
            lNewImportedMaterial.mTextures.mMetallicRoughnessTexture =
                PackMetalRoughTexture( lMaterial.metallic_texname, lMaterial.roughness_texname );
            lNewImportedMaterial.mTextures.mNormalTexture   = RetrieveTextureData( lMaterial.normal_texname );
            lNewImportedMaterial.mTextures.mEmissiveTexture = RetrieveTextureData( lMaterial.emissive_texname );

            lNewImportedMaterial.mAlpha.mMode = sImportedMaterial::AlphaMode::BLEND_MODE;

            mMaterials.push_back( lNewImportedMaterial );
            mMaterialIDLookup[lMaterialIndex++] = mMaterials.size() - 1;
        }

        for( auto const &lShape : lObjShapes )
        {
            std::set<int> lMaterialIDs( lShape.mesh.material_ids.begin(), lShape.mesh.material_ids.end() );

            for( int lMaterialID : lMaterialIDs )
            {
                sImportedMesh lMesh{};
                lMesh.mName       = lShape.name.empty()
                                        ? fmt::format( "UNNAMED_SHAPE_{}", mMaterials[mMaterialIDLookup[lMaterialID]].mName )
                                        : fmt::format( "{}_{}", lShape.name, mMaterials[mMaterialIDLookup[lMaterialID]].mName );
                lMesh.mMaterialID = static_cast<uint32_t>( mMaterialIDLookup[lMaterialID] );
                SE::Logging::Info( "Loading mesh: {}", lMesh.mName );

                for( int lFaceID = 0; lFaceID < lShape.mesh.material_ids.size(); lFaceID++ )
                {
                    if( lShape.mesh.material_ids[lFaceID] != lMaterialID )
                        continue;
                    AddVertex( lMesh, lShape.mesh.indices[3 * lFaceID + 0] );
                    AddVertex( lMesh, lShape.mesh.indices[3 * lFaceID + 1] );
                    AddVertex( lMesh, lShape.mesh.indices[3 * lFaceID + 2] );
                }

                if( !lMesh.mPositions.empty() )
                    mMeshes.push_back( lMesh );
            }
        }
    }

} // namespace SE::Core