#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"

// std
#include <set>

namespace std
{
    inline bool operator<( const tinyobj::index_t &a, const tinyobj::index_t &b )
    {
        if( a.vertex_index < b.vertex_index ) return true;
        if( a.vertex_index > b.vertex_index ) return false;

        if( a.normal_index < b.normal_index ) return true;
        if( a.normal_index > b.normal_index ) return false;

        if( a.texcoord_index < b.texcoord_index ) return true;
        if( a.texcoord_index > b.texcoord_index ) return false;

        return false;
    }
} // namespace std

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{

    /*! find vertex with given position, normal, texcoord, and return
        its vertex ID, or, if it doesn't exit, add it to the aMesh, and
        its just-created index */
    int AddVertex( TriangleMesh *aMesh, tinyobj::attrib_t &aAttributes, const tinyobj::index_t &aIdx,
                   std::map<tinyobj::index_t, int> &aKnownVertices )
    {
        if( aKnownVertices.find( aIdx ) != aKnownVertices.end() ) return aKnownVertices[aIdx];

        const math::vec3 *lVertexArray   = (const math::vec3 *)aAttributes.vertices.data();
        const math::vec3 *lNormalArray   = (const math::vec3 *)aAttributes.normals.data();
        const math::vec2 *lTexCoordArray = (const math::vec2 *)aAttributes.texcoords.data();

        int lNewID           = (int)aMesh->mVertex.size();
        aKnownVertices[aIdx] = lNewID;

        aMesh->mVertex.push_back( lVertexArray[aIdx.vertex_index] );
        if( aIdx.normal_index >= 0 )
        {
            while( aMesh->mNormal.size() < aMesh->mVertex.size() ) aMesh->mNormal.push_back( lNormalArray[aIdx.normal_index] );
        }
        if( aIdx.texcoord_index >= 0 )
        {
            while( aMesh->mTexCoord.size() < aMesh->mVertex.size() ) aMesh->mTexCoord.push_back( lTexCoordArray[aIdx.texcoord_index] );
        }

        if( aMesh->mTexCoord.size() > 0 ) aMesh->mTexCoord.resize( aMesh->mVertex.size() );
        if( aMesh->mNormal.size() > 0 ) aMesh->mNormal.resize( aMesh->mVertex.size() );

        return lNewID;
    }

    /*! load a texture (if not already loaded), and return its ID in the
        model's textures[] vector. Textures that could not get loaded
        return -1 */
    int loadTexture( Model *aModel, std::map<std::string, int> &aKnownTextures, const std::string &aInFileName,
                     const std::string &aModelPath )
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

    Model *loadOBJ( const std::string &lObjFile )
    {
        Model *lModel = new Model;

        const std::string lModelDir = lObjFile.substr( 0, lObjFile.rfind( '/' ) + 1 );

        tinyobj::attrib_t                aAttributes;
        std::vector<tinyobj::shape_t>    lShapes;
        std::vector<tinyobj::material_t> lMaterials;
        std::string                      lErrorString = "";

        bool lReadOK =
            tinyobj::LoadObj( &aAttributes, &lShapes, &lMaterials, &lErrorString, &lErrorString, lObjFile.c_str(), lModelDir.c_str(),
                              /* triangulate */ true );
        if( !lReadOK )
        {
            throw std::runtime_error( "Could not read OBJ model from " + lObjFile + " : " + lErrorString );
        }

        if( lMaterials.empty() ) throw std::runtime_error( "could not parse materials ..." );

        std::cout << "Done loading obj file - found " << lShapes.size() << " shapes with " << lMaterials.size() << " materials"
                  << std::endl;
        std::map<std::string, int> lKnownTextures;
        for( int lShapeID = 0; lShapeID < (int)lShapes.size(); lShapeID++ )
        {
            tinyobj::shape_t &lShape = lShapes[lShapeID];

            std::set<int> lMaterialIDs;
            for( auto lFaceMatID : lShape.mesh.material_ids ) lMaterialIDs.insert( lFaceMatID );

            std::map<tinyobj::index_t, int> lKnownVertices;

            for( int lMaterialID : lMaterialIDs )
            {
                TriangleMesh *lMesh = new TriangleMesh;

                for( int lFaceID = 0; lFaceID < lShape.mesh.material_ids.size(); lFaceID++ )
                {
                    if( lShape.mesh.material_ids[lFaceID] != lMaterialID ) continue;
                    tinyobj::index_t lIdx0 = lShape.mesh.indices[3 * lFaceID + 0];
                    tinyobj::index_t lIdx1 = lShape.mesh.indices[3 * lFaceID + 1];
                    tinyobj::index_t lIdx2 = lShape.mesh.indices[3 * lFaceID + 2];

                    math::ivec3 lIdx( AddVertex( lMesh, aAttributes, lIdx0, lKnownVertices ),
                                      AddVertex( lMesh, aAttributes, lIdx1, lKnownVertices ),
                                      AddVertex( lMesh, aAttributes, lIdx2, lKnownVertices ) );
                    lMesh->mIndex.push_back( lIdx );
                    lMesh->mDiffuse = (const math::vec3 &)lMaterials[lMaterialID].diffuse;
                    lMesh->mDiffuseTextureID =
                        loadTexture( lModel, lKnownTextures, lMaterials[lMaterialID].diffuse_texname, lModelDir );
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
} // namespace osc
