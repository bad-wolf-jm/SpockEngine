#include "ObjImporter.h"
#include <iostream>

#include "Core/Logging.h"
#include "fmt/core.h"

namespace SE::Core
{

    /*! find vertex with given position, normal, texcoord, and return
        its vertex ID, or, if it doesn't exit, add it to the aMesh, and
        its just-created index */
    int ObjImporter::AddVertex( const tinyobj::index_t &aIdx )
    {
        if( mKnownVertices.find( aIdx ) != mKnownVertices.end() ) return mKnownVertices[aIdx];

        const math::vec3 *lVertexArray   = (const math::vec3 *)mAttributes.vertices.data();
        const math::vec3 *lNormalArray   = (const math::vec3 *)mAttributes.normals.data();
        const math::vec2 *lTexCoordArray = (const math::vec2 *)mAttributes.texcoords.data();

        int lNewID           = (int)aMesh->mVertex.size();
        mKnownVertices[aIdx] = lNewID;

        aMesh->mVertex.push_back( lVertexArray[aIdx.vertex_index] );
        if( aIdx.normal_index >= 0 )
            while( aMesh->mNormal.size() < aMesh->mVertex.size() ) aMesh->mNormal.push_back( lNormalArray[aIdx.normal_index] );

        if( aIdx.texcoord_index >= 0 )
            while( aMesh->mTexCoord.size() < aMesh->mVertex.size() ) aMesh->mTexCoord.push_back( lTexCoordArray[aIdx.texcoord_index] );

        if( aMesh->mTexCoord.size() > 0 ) aMesh->mTexCoord.resize( aMesh->mVertex.size() );
        if( aMesh->mNormal.size() > 0 ) aMesh->mNormal.resize( aMesh->mVertex.size() );

        return lNewID;
    }

    /*! load a texture (if not already loaded), and return its ID in the
        model's textures[] vector. Textures that could not get loaded
        return -1 */
    int ObjImporter::LoadTexture( const std::string &aInFileName )
    {
        if( aInFileName == "" ) return -1;

        if( aKnownTextures.find( aInFileName ) != aKnownTextures.end() ) return aKnownTextures[aInFileName];

        std::string lFileName = aInFileName;
        // first, fix backspaces:
        for( auto &c : lFileName )
            if( c == '\\' ) c = '/';
        lFileName = aModelPath + "/" + lFileName;

        math::ivec2    lResolution;
        int            lComp;
        unsigned char *lImage     = stbi_load( lFileName.c_str(), &lResolution.x, &lResolution.y, &lComp, STBI_rgb_alpha );
        int            lTextureID = -1;
        if( lImage )
        {
            lTextureID           = (int)aModel->mTextures.size();
            Texture *texture     = new Texture;
            texture->mResolution = lResolution;
            texture->mPixel      = (uint32_t *)lImage;

            /* iw - actually, it seems that stbi loads the pictures
               mirrored along the y axis - mirror them here */
            for( int y = 0; y < lResolution.y / 2; y++ )
            {
                uint32_t *line_y     = texture->mPixel + y * lResolution.x;
                uint32_t *mirrored_y = texture->mPixel + ( lResolution.y - 1 - y ) * lResolution.x;
                int       mirror_y   = lResolution.y - 1 - y;
                for( int x = 0; x < lResolution.x; x++ )
                {
                    std::swap( line_y[x], mirrored_y[x] );
                }
            }

            aModel->mTextures.push_back( texture );
        }
        else
        {
            std::cout << GDT_TERMINAL_RED << "Could not load texture from " << lFileName << "!" << GDT_TERMINAL_DEFAULT << std::endl;
        }

        aKnownTextures[aInFileName] = lTextureID;
        return lTextureID;
    }

    ObjImporter::ObjImporter( const fs::path &aObjFile )
    {
        std::vector<tinyobj::shape_t>    mShapes;
        std::vector<tinyobj::material_t> mMaterials;

        std::string lErrorString = "";
        bool        lReadOK = tinyobj::LoadObj( &mAttributes, &mShapes, &mMaterials, &lErrorString, &lErrorString, aObjFile.c_str(),
                                                mModelDir.c_str(), true );
        if( !lReadOK ) throw std::runtime_error( "Could not read OBJ model from " + aObjFile + " : " + lErrorString );

        if( mMaterials.empty() ) throw std::runtime_error( "could not parse materials ..." );

        std::cout << "Done loading obj file - found " << mShapes.size() << " shapes with " << mMaterials.size() << " materials"
                  << std::endl;

        for( auto const &lShape : mShapes )
        {
            std::set<int> lMaterialIDs( lShape.mesh.material_ids.begin(), lShape.mesh.material_ids.end() );

            for( int lMaterialID : lMaterialIDs )
            {
                sImportedMesh lMesh{};

                for( int lFaceID = 0; lFaceID < lShape.mesh.material_ids.size(); lFaceID++ )
                {
                    if( lShape.mesh.material_ids[lFaceID] != lMaterialID ) continue;
                    tinyobj::index_t lIdx0 = lShape.mesh.indices[3 * lFaceID + 0];
                    tinyobj::index_t lIdx1 = lShape.mesh.indices[3 * lFaceID + 1];
                    tinyobj::index_t lIdx2 = lShape.mesh.indices[3 * lFaceID + 2];

                    // math::ivec3 lIdx( AddVertex( lIdx0 ), AddVertex( lIdx1 ), AddVertex( lIdx2 ) );
                    lMesh->mIndices.push_back( AddVertex( lIdx0 ) );
                    lMesh->mIndices.push_back( AddVertex( lIdx1 ) );
                    lMesh->mIndices.push_back( AddVertex( lIdx2 ) );
                    lMesh->mDiffuse          = (const math::vec3 &)mMaterials[lMaterialID].diffuse;
                    lMesh->mDiffuseTextureID = loadTexture( mMaterials[lMaterialID].diffuse_texname, mModelDir );
                }

                if( lMesh->mVertex.empty() )
                    delete lMesh;
                else
                    lModel->mMeshes.push_back( lMesh );
            }
        }

        // of course, you should be using tbb::parallel_for for stuff
        // like this:
        for( auto lMesh : lModel->mMeshes )
            for( auto lVtx : lMesh->mVertex ) lModel->mBounds.extend( lVtx );

        std::cout << "created a total of " << lModel->mMeshes.size() << " meshes" << std::endl;
        return lModel;
    }

} // namespace SE::Core