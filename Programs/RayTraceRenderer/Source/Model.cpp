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
        its vertex ID, or, if it doesn't exit, add it to the mesh, and
        its just-created index */
    int addVertex( TriangleMesh *mesh, tinyobj::attrib_t &attributes, const tinyobj::index_t &idx,
                   std::map<tinyobj::index_t, int> &knownVertices )
    {
        if( knownVertices.find( idx ) != knownVertices.end() ) return knownVertices[idx];

        const math::vec3 *vertex_array   = (const math::vec3 *)attributes.vertices.data();
        const math::vec3 *normal_array   = (const math::vec3 *)attributes.normals.data();
        const math::vec2 *texcoord_array = (const math::vec2 *)attributes.texcoords.data();

        int newID          = (int)mesh->mVertex.size();
        knownVertices[idx] = newID;

        mesh->mVertex.push_back( vertex_array[idx.vertex_index] );
        if( idx.normal_index >= 0 )
        {
            while( mesh->mNormal.size() < mesh->mVertex.size() ) mesh->mNormal.push_back( normal_array[idx.normal_index] );
        }
        if( idx.texcoord_index >= 0 )
        {
            while( mesh->mTexCoord.size() < mesh->mVertex.size() ) mesh->mTexCoord.push_back( texcoord_array[idx.texcoord_index] );
        }

        // just for sanity's sake:
        if( mesh->mTexCoord.size() > 0 ) mesh->mTexCoord.resize( mesh->mVertex.size() );
        // just for sanity's sake:
        if( mesh->mNormal.size() > 0 ) mesh->mNormal.resize( mesh->mVertex.size() );

        return newID;
    }

    /*! load a texture (if not already loaded), and return its ID in the
        model's textures[] vector. Textures that could not get loaded
        return -1 */
    int loadTexture( Model *model, std::map<std::string, int> &knownTextures, const std::string &inFileName,
                     const std::string &modelPath )
    {
        if( inFileName == "" ) return -1;

        if( knownTextures.find( inFileName ) != knownTextures.end() ) return knownTextures[inFileName];

        std::string fileName = inFileName;
        // first, fix backspaces:
        for( auto &c : fileName )
            if( c == '\\' ) c = '/';
        fileName = modelPath + "/" + fileName;

        math::ivec2    res;
        int            comp;
        unsigned char *image     = stbi_load( fileName.c_str(), &res.x, &res.y, &comp, STBI_rgb_alpha );
        int            textureID = -1;
        if( image )
        {
            textureID            = (int)model->mTextures.size();
            Texture *texture     = new Texture;
            texture->mResolution = res;
            texture->mPixel      = (uint32_t *)image;

            /* iw - actually, it seems that stbi loads the pictures
               mirrored along the y axis - mirror them here */
            for( int y = 0; y < res.y / 2; y++ )
            {
                uint32_t *line_y     = texture->mPixel + y * res.x;
                uint32_t *mirrored_y = texture->mPixel + ( res.y - 1 - y ) * res.x;
                int       mirror_y   = res.y - 1 - y;
                for( int x = 0; x < res.x; x++ )
                {
                    std::swap( line_y[x], mirrored_y[x] );
                }
            }

            model->mTextures.push_back( texture );
        }
        else
        {
            std::cout << GDT_TERMINAL_RED << "Could not load texture from " << fileName << "!" << GDT_TERMINAL_DEFAULT << std::endl;
        }

        knownTextures[inFileName] = textureID;
        return textureID;
    }

    Model *loadOBJ( const std::string &objFile )
    {
        Model *model = new Model;

        const std::string modelDir = objFile.substr( 0, objFile.rfind( '/' ) + 1 );

        tinyobj::attrib_t                attributes;
        std::vector<tinyobj::shape_t>    shapes;
        std::vector<tinyobj::material_t> materials;
        std::string                      err = "";

        bool readOK = tinyobj::LoadObj( &attributes, &shapes, &materials, &err, &err, objFile.c_str(), modelDir.c_str(),
                                        /* triangulate */ true );
        if( !readOK )
        {
            throw std::runtime_error( "Could not read OBJ model from " + objFile + " : " + err );
        }

        if( materials.empty() ) throw std::runtime_error( "could not parse materials ..." );

        std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials"
                  << std::endl;
        std::map<std::string, int> knownTextures;
        for( int shapeID = 0; shapeID < (int)shapes.size(); shapeID++ )
        {
            tinyobj::shape_t &shape = shapes[shapeID];

            std::set<int> materialIDs;
            for( auto faceMatID : shape.mesh.material_ids ) materialIDs.insert( faceMatID );

            std::map<tinyobj::index_t, int> knownVertices;

            for( int materialID : materialIDs )
            {
                TriangleMesh *mesh = new TriangleMesh;

                for( int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++ )
                {
                    if( shape.mesh.material_ids[faceID] != materialID ) continue;
                    tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                    tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                    tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                    math::ivec3 idx( addVertex( mesh, attributes, idx0, knownVertices ),
                                     addVertex( mesh, attributes, idx1, knownVertices ),
                                     addVertex( mesh, attributes, idx2, knownVertices ) );
                    mesh->mIndex.push_back( idx );
                    mesh->mDiffuse          = (const math::vec3 &)materials[materialID].diffuse;
                    mesh->mDiffuseTextureID = loadTexture( model, knownTextures, materials[materialID].diffuse_texname, modelDir );
                }

                if( mesh->mVertex.empty() )
                    delete mesh;
                else
                    model->mMeshes.push_back( mesh );
            }
        }

        // of course, you should be using tbb::parallel_for for stuff
        // like this:
        for( auto mesh : model->mMeshes )
            for( auto vtx : mesh->mVertex ) model->mBounds.extend( vtx );

        std::cout << "created a total of " << model->mMeshes.size() << " meshes" << std::endl;
        return model;
    }
} // namespace osc
