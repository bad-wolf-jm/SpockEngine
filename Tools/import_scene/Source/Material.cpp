#include "Material.h"

// #include "CoreEngine/GraphicContext/Imageloader.h"
#include "Conversion.h"
#include "Core/Logging.h"


namespace fs = std::filesystem;

#define GET_MATERIAL_COLOR_3D( material, key, result )                                                                                                                             \
    do                                                                                                                                                                             \
    {                                                                                                                                                                              \
        aiColor3D l_Value{ result.x, result.y, result.z };                                                                                                                         \
        auto ret = material->Get( key, l_Value );                                                                                                                                  \
        result   = to_vec3( l_Value );                                                                                                                                             \
    } while( 0 )

#define GET_MATERIAL_FLOAT( material, key, result )                                                                                                                                \
    do                                                                                                                                                                             \
    {                                                                                                                                                                              \
        auto ret = material->Get( key, result );                                                                                                                                   \
    } while( 0 )

#define GET_MATERIAL_INT( material, key, result )                                                                                                                                  \
    do                                                                                                                                                                             \
    {                                                                                                                                                                              \
        auto ret = material->Get( key, result );                                                                                                                                   \
    } while( 0 )

#define GET_MATERIAL_BOOL( material, key, result )                                                                                                                                 \
    do                                                                                                                                                                             \
    {                                                                                                                                                                              \
        int l_Value;                                                                                                                                                               \
        auto ret = material->Get( key, l_Value );                                                                                                                                  \
        result   = ( l_Value != 0 );                                                                                                                                               \
    } while( 0 )

static const char *TextureTypeToString( aiTextureType in )
{
    switch( in )
    {
    case aiTextureType_NONE:
        return "n/a";
    case aiTextureType_DIFFUSE:
        return "Diffuse";
    case aiTextureType_SPECULAR:
        return "Specular";
    case aiTextureType_AMBIENT:
        return "Ambient";
    case aiTextureType_EMISSIVE:
        return "Emissive";
    case aiTextureType_OPACITY:
        return "Opacity";
    case aiTextureType_NORMALS:
        return "Normals";
    case aiTextureType_HEIGHT:
        return "Height";
    case aiTextureType_SHININESS:
        return "Shininess";
    case aiTextureType_DISPLACEMENT:
        return "Displacement";
    case aiTextureType_LIGHTMAP:
        return "Lightmap";
    case aiTextureType_REFLECTION:
        return "Reflection";
    case aiTextureType_BASE_COLOR:
        return "BaseColor";
    case aiTextureType_NORMAL_CAMERA:
        return "NormalCamera";
    case aiTextureType_EMISSION_COLOR:
        return "EmissionColor";
    case aiTextureType_METALNESS:
        return "Metalness";
    case aiTextureType_DIFFUSE_ROUGHNESS:
        return "DiffuseRoughness";
    case aiTextureType_AMBIENT_OCCLUSION:
        return "AmbientOcclusion";
    case aiTextureType_UNKNOWN:
        return "Unknown";
    default:
        break;
    }
    return "BUG";
}

static std::string removeLeadingCharacters(std::string str, const char charToRemove) {
    std::string x = str;
    x.erase(0, std::min(x.find_first_not_of(charToRemove), x.size() - 1));
    return x;
}

static bool LoadTexture( fs::path a_AssetRoot, const aiMaterial *a_Material, const aiTextureType a_Type, MaterialData &o_Material )
{
    aiString l_TexturePath;
    aiTextureMapping l_TextureMapping;
    aiTextureMapMode l_TextureMapMode[2];
    uint32_t l_UVIndex = 0;

    if( a_Material->GetTextureCount( a_Type ) == 0 )
        return false;

    a_Material->GetTexture( a_Type, 0, &l_TexturePath, &l_TextureMapping, &l_UVIndex, nullptr, nullptr, l_TextureMapMode );

    LTSE::Logging::Info( "    {} - {}", TextureTypeToString( a_Type ), a_Material->GetTextureCount( a_Type ) );
    LTSE::Logging::Info( "     - root: {}", a_AssetRoot.string() );
    LTSE::Logging::Info( "     - path: {}", ( a_AssetRoot / removeLeadingCharacters(std::string( l_TexturePath.C_Str() ), '/' )).string() );
    LTSE::Logging::Info( "     - ipath: {}", removeLeadingCharacters(std::string( l_TexturePath.C_Str() ), '/' ) );
    LTSE::Logging::Info( "     - mapping: {}", l_TextureMapping );
    LTSE::Logging::Info( "     - map_mode=({}, {})", l_TextureMapMode[0], l_TextureMapMode[1] );

    o_Material.TextureFlags |= ( 1 << (size_t)a_Type );
    o_Material.TexturePaths[a_Type] = ( a_AssetRoot / removeLeadingCharacters(std::string( l_TexturePath.C_Str() ), '/' ) ).string();
    o_Material.TexCoords[a_Type]    = l_UVIndex;
    return true;
}

#define AI_MATKEY_METALLIC_FACTOR "$mat.metallicFactor", 0, 0
#define AI_MATKEY_ROUGHNESS_FACTOR "$mat.roughnessFactor", 0, 0



bool LoadMaterials( const aiScene *a_SceneData, fs::path a_AssetRoot, std::vector<MaterialData> &a_Materials )
{
    LTSE::Logging::Info( "Loading materials..." );
    a_Materials.resize( a_SceneData->mNumMaterials );
    for( auto l_MaterialIdx = 0; l_MaterialIdx < a_SceneData->mNumMaterials; l_MaterialIdx++ )
    {
        auto l_Material = a_SceneData->mMaterials[l_MaterialIdx];
        MaterialData l_NewMaterial{};
        l_NewMaterial.Name = l_Material->GetName().C_Str();
        l_NewMaterial.ID   = l_MaterialIdx;
        if( l_NewMaterial.Name.length() == 0 )
            l_NewMaterial.Name = fmt::format( "Unnamed_material_{}", l_MaterialIdx );
        LTSE::Logging::Info( "  #{} - {}", l_MaterialIdx, l_NewMaterial.Name );

        l_NewMaterial.DiffuseColor = math::vec3{ 0.0f, 0.0f, 0.0f };
        GET_MATERIAL_COLOR_3D( l_Material, AI_MATKEY_COLOR_DIFFUSE, l_NewMaterial.DiffuseColor );

        l_NewMaterial.SpecularColor = math::vec3{ 0.0f, 0.0f, 0.0f };
        GET_MATERIAL_COLOR_3D( l_Material, AI_MATKEY_COLOR_SPECULAR, l_NewMaterial.SpecularColor );

        l_NewMaterial.AmbientColor = math::vec3{ 0.0f, 0.0f, 0.0f };
        GET_MATERIAL_COLOR_3D( l_Material, AI_MATKEY_COLOR_AMBIENT, l_NewMaterial.AmbientColor );
        if( l_NewMaterial.AmbientColor == math::vec3{ 0.0f, 0.0f, 0.0f } )
            l_NewMaterial.AmbientColor = math::vec3{ 1.0f, 1.0f, 1.0f };

        l_NewMaterial.EmissiveColor = math::vec3{ 0.0f, 0.0f, 0.0f };
        GET_MATERIAL_COLOR_3D( l_Material, AI_MATKEY_COLOR_EMISSIVE, l_NewMaterial.EmissiveColor );

        l_NewMaterial.TransparentColor = math::vec3{ 0.0f, 0.0f, 0.0f };
        GET_MATERIAL_COLOR_3D( l_Material, AI_MATKEY_COLOR_TRANSPARENT, l_NewMaterial.TransparentColor );

        l_NewMaterial.ReflectiveColor = math::vec3{ 0.0f, 0.0f, 0.0f };
        GET_MATERIAL_COLOR_3D( l_Material, AI_MATKEY_COLOR_REFLECTIVE, l_NewMaterial.ReflectiveColor );

        GET_MATERIAL_BOOL( l_Material, AI_MATKEY_ENABLE_WIREFRAME, l_NewMaterial.IsWireframe );
        GET_MATERIAL_BOOL( l_Material, AI_MATKEY_TWOSIDED, l_NewMaterial.IsTwoSided );

        GET_MATERIAL_INT( l_Material, AI_MATKEY_BLEND_FUNC, l_NewMaterial.BlendFunction );

        GET_MATERIAL_FLOAT( l_Material, AI_MATKEY_REFLECTIVITY, l_NewMaterial.Reflectivity );
        GET_MATERIAL_FLOAT( l_Material, AI_MATKEY_OPACITY, l_NewMaterial.Opacity );
        GET_MATERIAL_FLOAT( l_Material, AI_MATKEY_SHININESS, l_NewMaterial.Shininess );
        GET_MATERIAL_FLOAT( l_Material, AI_MATKEY_SHININESS_STRENGTH, l_NewMaterial.ShininessStrength );
        GET_MATERIAL_FLOAT( l_Material, AI_MATKEY_REFRACTI, l_NewMaterial.RefractionIndex );

        GET_MATERIAL_FLOAT( l_Material, AI_MATKEY_METALLIC_FACTOR, l_NewMaterial.MetallicFactor );
        GET_MATERIAL_FLOAT( l_Material, AI_MATKEY_ROUGHNESS_FACTOR, l_NewMaterial.RoughnessFactor );

        l_NewMaterial.TexturePaths.resize( AI_TEXTURE_TYPE_MAX + 1 );
        l_NewMaterial.TexCoords.resize( AI_TEXTURE_TYPE_MAX + 1 );
        l_NewMaterial.TextureFlags = 0x0;

        LoadTexture( a_AssetRoot, l_Material, aiTextureType_NONE, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_AMBIENT, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_DIFFUSE, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_SPECULAR, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_EMISSIVE, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_SHININESS, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_OPACITY, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_DISPLACEMENT, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_AMBIENT_OCCLUSION, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_LIGHTMAP, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_DIFFUSE_ROUGHNESS, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_HEIGHT, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_NORMALS, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_BASE_COLOR, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_NORMAL_CAMERA, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_EMISSION_COLOR, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_METALNESS, l_NewMaterial );
        LoadTexture( a_AssetRoot, l_Material, aiTextureType_UNKNOWN, l_NewMaterial );

        a_Materials[l_MaterialIdx] = l_NewMaterial;
    }
    return true;
}
