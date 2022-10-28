#include "Importer.h"

#include <fstream>
#include <iterator>
#include <unordered_map>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <fstream>

#include "Core/Logging.h"
#include "Core/Math/Types.h"
#include "Core/Types.h"

#include "Graphics/API/ColorFormat.h"
#include "Graphics/API/TextureTypes.h"

#include "yaml-cpp/yaml.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace LTSE::Core;

static void PrintNodeTree( int level, NodeData &CurrentNode, std::vector<NodeData> o_NodesList )
{
    fmt::print( "{:{}}", "", level * 2 );
    fmt::print( "{}\n", CurrentNode.Name );
    for( auto &c : CurrentNode.ChildrenID )
        PrintNodeTree( level + 1, o_NodesList[c], o_NodesList );
}

static inline YAML::Emitter &EmitColor( YAML::Emitter &out, math::vec3 a_Color )
{
    out << YAML::Flow;
    out << YAML::BeginMap;
    out << YAML::Key << "r" << YAML::Value << a_Color.x;
    out << YAML::Key << "g" << YAML::Value << a_Color.y;
    out << YAML::Key << "b" << YAML::Value << a_Color.z;
    out << YAML::EndMap;
    return out;
}

static inline YAML::Emitter &EmitVector( YAML::Emitter &out, math::vec3 a_Color )
{
    out << YAML::Flow;
    out << YAML::BeginMap;
    out << YAML::Key << "x" << YAML::Value << a_Color.x;
    out << YAML::Key << "y" << YAML::Value << a_Color.y;
    out << YAML::Key << "z" << YAML::Value << a_Color.z;
    out << YAML::EndMap;
    return out;
}

static inline YAML::Emitter &EmitQuaternion( YAML::Emitter &out, math::quat a_Color )
{
    out << YAML::Flow;
    out << YAML::BeginMap;
    out << YAML::Key << "x" << YAML::Value << a_Color.x;
    out << YAML::Key << "y" << YAML::Value << a_Color.y;
    out << YAML::Key << "z" << YAML::Value << a_Color.z;
    out << YAML::Key << "w" << YAML::Value << a_Color.w;
    out << YAML::EndMap;
    return out;
}

static inline YAML::Emitter &EmitColor( YAML::Emitter &out, math::vec4 a_Color )
{
    out << YAML::Flow;
    out << YAML::BeginMap;
    out << YAML::Key << "r" << YAML::Value << a_Color.x;
    out << YAML::Key << "g" << YAML::Value << a_Color.y;
    out << YAML::Key << "b" << YAML::Value << a_Color.z;
    out << YAML::Key << "a" << YAML::Value << a_Color.w;
    out << YAML::EndMap;
    return out;
}

static inline YAML::Emitter &EmitKeyFrame( YAML::Emitter &out, KeyFrame<math::vec3> a_Frame )
{
    out << YAML::Flow;
    out << YAML::BeginMap;
    out << YAML::Key << "tick" << YAML::Value << a_Frame.Tick;
    out << YAML::Key << "value" << YAML::Value;
    EmitVector( out, a_Frame.Value );
    out << YAML::EndMap;
    return out;
}

static inline YAML::Emitter &EmitKeyFrame( YAML::Emitter &out, KeyFrame<math::quat> a_Frame )
{
    out << YAML::Flow;
    out << YAML::BeginMap;
    out << YAML::Key << "tick" << YAML::Value << a_Frame.Tick;
    out << YAML::Key << "value" << YAML::Value;
    EmitQuaternion( out, a_Frame.Value );
    out << YAML::EndMap;
    return out;
}

struct TextureTypeMap
{
    TextureType Type;
    char *Name;
};

// clang-format off
static constexpr TextureTypeMap g_TextureTypes[] = {
    { TextureType::DIFFUSE,           "BASE" },
    { TextureType::SPECULAR,          "SPECULAR" },
    { TextureType::AMBIENT,           "AMBIENT" },
    { TextureType::EMISSIVE,          "EMISSIVE" },
    { TextureType::OPACITY,           "OPACITY" },
    { TextureType::NORMALS,           "NORMALS" },
    { TextureType::HEIGHT,            "HEIGHT" },
    { TextureType::SHININESS,         "SHININESS" },
    { TextureType::DISPLACEMENT,      "DISPLACEMENT" },
    { TextureType::LIGHTMAP,          "AO" },
    { TextureType::REFLECTION,        "REFLECTION" },
    { TextureType::BASE_COLOR,        "BASE_COLOR" },
    { TextureType::NORMAL_CAMERA,     "NORMAL_CAMERA" },
    { TextureType::EMISSION_COLOR,    "EMISSION_COLOR" },
    { TextureType::METALNESS,         "METALNESS" },
    { TextureType::DIFFUSE_ROUGHNESS, "DIFFUSE_ROUGHNESS" },
    { TextureType::AMBIENT_OCCLUSION, "AMBIENT_OCCLUSION" },
    { TextureType::UNKNOWN,           "AO_METALLIC_ROUGHNESS" }
};
// clang-format on

void EmitTextureSpecification( YAML::Emitter &out, fs::path a_MaterialRoot, fs::path a_TextureRoot, MaterialData &a_MaterialData, std::string a_Name, TextureType a_Type )
{
    if( ( a_Type == TextureType::METALNESS ) && !( a_MaterialData.TextureFlags & ( 1 << a_Type ) ) && ( a_MaterialData.TextureFlags & ( 1 << TextureType::UNKNOWN ) ) )
    {
        const std::string l_TexturePath      = a_MaterialData.TexturePaths[TextureType::UNKNOWN];
        fs::path l_TargetTextureFileName     = fmt::format( "{}_metal.png", fs::path( l_TexturePath ).filename().stem().string() );
        fs::path l_TargetTextureRelativePath = a_TextureRoot / l_TargetTextureFileName;

        sImageData l_TextureImage = LoadImageData( fs::path( l_TexturePath ) );

        std::vector<uint8_t> l_ChannelData( l_TextureImage.mWidth * l_TextureImage.mHeight * sizeof( uint8_t ) );

        for( uint32_t y = 0; y < l_TextureImage.mHeight; y++ )
        {
            for( uint32_t x = 0; x < l_TextureImage.mWidth; x++ )
            {
                uint8_t l_SourceStride = 0;
                switch( l_TextureImage.mFormat )
                {
                case eColorFormat::R8_UNORM:
                    l_SourceStride = 1;
                    break;
                case eColorFormat::RG8_UNORM:
                    l_SourceStride = 2;
                    break;
                case eColorFormat::RGB8_UNORM:
                    l_SourceStride = 3;
                    break;
                case eColorFormat::RGBA8_UNORM:
                    l_SourceStride = 4;
                    break;
                default:
                    break;
                }
                uint8_t l_PixelValue                         = l_TextureImage.mPixelData[y * l_TextureImage.mWidth * l_SourceStride + x * l_SourceStride + 2];
                l_ChannelData[y * l_TextureImage.mWidth + x] = l_PixelValue;
            }
        }
        stbi_write_png( ( a_MaterialRoot / l_TargetTextureRelativePath ).string().c_str(), l_TextureImage.mWidth, l_TextureImage.mHeight, 1, l_ChannelData.data(),
                        l_TextureImage.mWidth );
        out << YAML::Key << a_Name << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "file_path" << YAML::Value << l_TargetTextureRelativePath.string();
        out << YAML::Key << "minification" << YAML::Value << "linear";
        out << YAML::Key << "magnification" << YAML::Value << "linear";
        out << YAML::Key << "mipmap" << YAML::Value << "linear";
        out << YAML::Key << "wrapping_mode" << YAML::Value << "repeat";
        out << YAML::EndMap;
    }
    else if( ( a_Type == TextureType::DIFFUSE_ROUGHNESS ) && !( a_MaterialData.TextureFlags & ( 1 << a_Type ) ) && ( a_MaterialData.TextureFlags & ( 1 << TextureType::UNKNOWN ) ) )
    {
        const std::string l_TexturePath      = a_MaterialData.TexturePaths[TextureType::UNKNOWN];
        fs::path l_TargetTextureFileName     = fmt::format( "{}_rough.png", fs::path( l_TexturePath ).filename().stem().string() );
        fs::path l_TargetTextureRelativePath = a_TextureRoot / l_TargetTextureFileName;

        sImageData l_TextureImage = LoadImageData( fs::path( l_TexturePath ) );

        std::vector<uint8_t> l_ChannelData( l_TextureImage.mWidth * l_TextureImage.mHeight * sizeof( uint8_t ) );

        for( uint32_t y = 0; y < l_TextureImage.mHeight; y++ )
        {
            for( uint32_t x = 0; x < l_TextureImage.mWidth; x++ )
            {
                uint8_t l_SourceStride = 0;
                switch( l_TextureImage.mFormat )
                {
                case eColorFormat::R8_UNORM:
                    l_SourceStride = 1;
                    break;
                case eColorFormat::RG8_UNORM:
                    l_SourceStride = 2;
                    break;
                case eColorFormat::RGB8_UNORM:
                    l_SourceStride = 3;
                    break;
                case eColorFormat::RGBA8_UNORM:
                    l_SourceStride = 4;
                    break;
                default:
                    break;
                }
                uint8_t l_PixelValue                         = l_TextureImage.mPixelData[y * l_TextureImage.mWidth * l_SourceStride + x * l_SourceStride + 1];
                l_ChannelData[y * l_TextureImage.mWidth + x] = l_PixelValue;
            }
        }
        stbi_write_png( ( a_MaterialRoot / l_TargetTextureRelativePath ).string().c_str(), l_TextureImage.mWidth, l_TextureImage.mHeight, 1, l_ChannelData.data(),
                        l_TextureImage.mWidth );
        out << YAML::Key << a_Name << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "file_path" << YAML::Value << l_TargetTextureRelativePath.string();
        out << YAML::Key << "minification" << YAML::Value << "linear";
        out << YAML::Key << "magnification" << YAML::Value << "linear";
        out << YAML::Key << "mipmap" << YAML::Value << "linear";
        out << YAML::Key << "wrapping_mode" << YAML::Value << "repeat";
        out << YAML::EndMap;
    }
    else if( !( a_MaterialData.TextureFlags & ( 1 << a_Type ) ) )
    {
        out << YAML::Key << a_Name << YAML::Value << YAML::Null;
        return;
    }
    else
    {
        std::string l_TexturePath            = a_MaterialData.TexturePaths[a_Type];
        fs::path l_TextureFileName           = fs::path( l_TexturePath ).filename();
        fs::path l_TargetTextureRelativePath = a_TextureRoot / l_TextureFileName;

        if( !fs::exists( a_MaterialRoot / l_TargetTextureRelativePath ) )
        {
            std::cout << fs::path( l_TexturePath ) << "====>" << l_TargetTextureRelativePath << std::endl;
            fs::copy_file( fs::path( l_TexturePath ), a_MaterialRoot / l_TargetTextureRelativePath );
        }

        out << YAML::Key << a_Name << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "file_path" << YAML::Value << l_TargetTextureRelativePath.string();
        out << YAML::Key << "minification" << YAML::Value << "linear";
        out << YAML::Key << "magnification" << YAML::Value << "linear";
        out << YAML::Key << "mipmap" << YAML::Value << "linear";
        out << YAML::Key << "wrapping_mode" << YAML::Value << "repeat";
        out << YAML::EndMap;
    }
}

void EmitMaterial( fs::path a_MaterialRoot, MaterialData &a_MaterialData )
{
    fs::path l_MaterialPath         = a_MaterialRoot / a_MaterialData.Name;
    fs::path l_MaterialTexturesPath = l_MaterialPath / "textures";
    fs::path l_MaterialConfigFile   = l_MaterialPath / "material.yaml";
    fs::create_directories( l_MaterialTexturesPath );

    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "asset";
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << a_MaterialData.Name;
    out << YAML::Key << "type" << YAML::Value << "material";
    out << YAML::Key << "data";
    out << YAML::BeginMap;
    out << YAML::Key << "shading" << YAML::Value << YAML::BeginMap;
    {
        out << YAML::Key << "two_sided" << YAML::Value << a_MaterialData.IsTwoSided;
        out << YAML::Key << "use_alpha_mask" << YAML::Value << false;
        out << YAML::Key << "line_width" << YAML::Value << 1.0f;
        out << YAML::Key << "alpha_mask_threshold" << YAML::Value << 0.5f;
    }
    out << YAML::EndMap;
    out << YAML::Key << "constants" << YAML::Value << YAML::BeginMap;
    {
        out << YAML::Key << "occlusion_strength" << YAML::Value << 0.0f;
        out << YAML::Key << "metallic_factor" << YAML::Value << a_MaterialData.MetallicFactor;
        out << YAML::Key << "roughness_factor" << YAML::Value << a_MaterialData.RoughnessFactor;
    }
    out << YAML::EndMap;
    out << YAML::Key << "colors" << YAML::Value << YAML::BeginMap;
    {
        out << YAML::Key << "albedo" << YAML::Value;
        EmitColor( out, math::vec4( a_MaterialData.AmbientColor, 1.0f ) );
        out << YAML::Key << "diffuse" << YAML::Value;
        EmitColor( out, math::vec4( a_MaterialData.DiffuseColor, 1.0f ) );
        out << YAML::Key << "specular" << YAML::Value;
        EmitColor( out, math::vec4( a_MaterialData.SpecularColor, 1.0f ) );
        out << YAML::Key << "emissive" << YAML::Value;
        EmitColor( out, math::vec4( a_MaterialData.EmissiveColor, 1.0f ) );
    }
    out << YAML::EndMap;
    out << YAML::Key << "textures" << YAML::Value << YAML::BeginMap;
    {
        EmitTextureSpecification( out, l_MaterialPath, "textures", a_MaterialData, "albedo", TextureType::DIFFUSE );
        EmitTextureSpecification( out, l_MaterialPath, "textures", a_MaterialData, "normals", TextureType::NORMALS );
        EmitTextureSpecification( out, l_MaterialPath, "textures", a_MaterialData, "occlusion", TextureType::LIGHTMAP );
        EmitTextureSpecification( out, l_MaterialPath, "textures", a_MaterialData, "emissive", TextureType::EMISSIVE );
        EmitTextureSpecification( out, l_MaterialPath, "textures", a_MaterialData, "metalness", TextureType::METALNESS );
        EmitTextureSpecification( out, l_MaterialPath, "textures", a_MaterialData, "roughness", TextureType::DIFFUSE_ROUGHNESS );
    }
    out << YAML::EndMap;
    out << YAML::EndMap;
    out << YAML::EndMap;
    out << YAML::EndMap;

    std::ofstream fout( l_MaterialConfigFile );
    fout << out.c_str();
}

void ReadAssetFile( const fs::path &a_FilePath, const fs::path &a_YamlFilePath )
{
    LTSE::Logging::Info( "Loading asset file: {}", a_FilePath.string() );

    if( !fs::exists( a_FilePath ) )
        return;

    auto l_AssetRoot       = a_FilePath.parent_path();
    auto l_AssetOutputRoot = a_YamlFilePath;

    // if (fs::exists(l_AssetOutputRoot)) return;

    Assimp::Importer importer;
    const aiScene *l_Scene         = nullptr;
    const uint32_t l_FileReadFlags = aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality |
                                     aiProcess_LimitBoneWeights | aiProcess_RemoveRedundantMaterials | aiProcess_SplitLargeMeshes | aiProcess_Triangulate | aiProcess_GenUVCoords |
                                     aiProcess_SortByPType | aiProcess_FindDegenerates | aiProcess_FindInvalidData;
    l_Scene = importer.ReadFile( a_FilePath.string(), l_FileReadFlags );

    if( !l_Scene )
    {
        LTSE::Logging::Error( "Could not load asset file \"{}\": {}", a_FilePath.string(), importer.GetErrorString() );
        return;
    }

    if( !l_Scene->mRootNode )
    {
        LTSE::Logging::Error( "Could not load asset file \"{}\": {}", a_FilePath.string(), importer.GetErrorString() );
        return;
    }

    LTSE::Logging::Info( "Successfully loaded asset file \"{}\"", a_FilePath.string() );
    LTSE::Logging::Info( "Asset info:" );
    LTSE::Logging::Info( " - Root: {}", l_AssetRoot.string() );
    LTSE::Logging::Info( " - {} meshes", l_Scene->mNumMeshes );
    LTSE::Logging::Info( " - {} embedded textures", l_Scene->mNumTextures );
    LTSE::Logging::Info( " - {} materials", l_Scene->mNumMaterials );
    LTSE::Logging::Info( " - {} cameras", l_Scene->mNumCameras );
    LTSE::Logging::Info( " - {} lights", l_Scene->mNumLights );
    LTSE::Logging::Info( " - {} animations", l_Scene->mNumAnimations );

    fs::create_directories( l_AssetOutputRoot / "meshes" );
    fs::create_directories( l_AssetOutputRoot / "materials" );
    fs::path l_AssetConfigFile = l_AssetOutputRoot / "asset.yaml";
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "general" << YAML::Value;
    {
        out << YAML::BeginMap;
        out << YAML::Key << "asset_name" << YAML::Value << "FOO";
        out << YAML::Key << "asset_version" << YAML::Value << "1.0.0";
        out << YAML::Key << "asset_origin" << YAML::Value << "NONE";
        out << YAML::EndMap;
    }

    std::unordered_map<std::string, std::string> l_TexturePathMap;
    std::vector<std::string> l_TexturePaths;

    if( l_Scene->mNumTextures > 0 && l_Scene->mTextures != nullptr )
        LTSE::Logging::Info( "Loading embedded textures" );

    std::vector<MaterialData> Materials;
    if( ( l_Scene->mNumMaterials > 0 ) && ( l_Scene->mMaterials != nullptr ) )
        LoadMaterials( l_Scene, l_AssetRoot, Materials );

    std::unordered_map<uint32_t, std::string> l_MaterialPathMap;
    for( auto &l_Material : Materials )
    {
        EmitMaterial( l_AssetOutputRoot / "materials", l_Material );
        l_MaterialPathMap[l_Material.ID] = fmt::format( "materials/{}/material.yaml", l_Material.Name );
    }

    std::vector<std::shared_ptr<MeshData>> Meshes;
    if( ( l_Scene->mNumMeshes > 0 ) && ( l_Scene->mMeshes != nullptr ) )
        LoadMeshes( l_Scene, Meshes, Materials );

    std::map<std::string, int32_t> l_Nodes;
    std::vector<NodeData> o_NodesList( CountNodes( l_Scene->mRootNode ) );
    int32_t l_NodeCount = LoadSceneNodes( 0, 0, l_Scene, l_Scene->mRootNode, NodeData{}, Meshes, l_Nodes, o_NodesList );

    std::map<std::string, int32_t> l_Bones;
    std::vector<BoneData> o_BonesList = {};

    for( auto &l_Node : o_NodesList )
    {
        if( l_Node.Meshes.size() == 0 )
            continue;

        for( auto &l_Mesh : l_Node.Meshes )
        {
            if( l_Mesh->Bones.size() == 0 )
                continue;

            for( auto &l_Bone : l_Mesh->Bones )
            {
                if( l_Bones.find( l_Bone.Name ) == l_Bones.end() )
                {
                    l_Bones[l_Bone.Name] = o_BonesList.size();
                    l_Bone.NodeID        = l_Nodes[l_Bone.Name];
                    o_BonesList.push_back( l_Bone );
                }
            }
        }
    }

    out << YAML::Key << "bones" << YAML::Value << YAML::BeginSeq;
    {
        for( auto &l_Bone : o_BonesList )
        {
            out << YAML::BeginMap;
            out << YAML::Key << "node_name" << YAML::Value << l_Bone.Name;
            out << YAML::Key << "node_id" << YAML::Value << l_Bone.NodeID;
            out << YAML::Key << "inverse_bind_matrix" << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for( int i = 0; i < 4; i++ )
                for( int j = 0; j < 4; j++ )
                    out << l_Bone.InverseBindMatrix[i][j];
            out << YAML::EndSeq;
            out << YAML::EndMap;
        }
    }
    out << YAML::EndSeq;

    out << YAML::Key << "meshes" << YAML::Value << YAML::BeginSeq;
    {
        for( auto &l_Mesh : Meshes )
        {
            auto &l_MeshName = l_Mesh->Name;

            fs::path l_MeshPath = fs::path( "meshes" ) / l_MeshName;
            if( !fs::exists( a_YamlFilePath / l_MeshPath ) )
                fs::create_directories( a_YamlFilePath / l_MeshPath );

            fs::path l_VertexBufferPath = a_YamlFilePath / l_MeshPath / "Vertices.dat";
            std::ofstream l_VertexBufferFile( l_VertexBufferPath, std::ios::out | std::ios::binary );
            size_t size = l_Mesh->Vertices.size();
            l_VertexBufferFile.write( (char *)&size, sizeof( size ) );
            l_VertexBufferFile.write( (char *)&l_Mesh->Vertices[0], l_Mesh->Vertices.size() * sizeof( LTSE::Core::VertexData ) );
            l_VertexBufferFile.flush();

            fs::path l_IndexBufferPath = a_YamlFilePath / l_MeshPath / "Indices.dat";
            std::ofstream l_IndexBufferFile( l_IndexBufferPath, std::ios::out | std::ios::binary );
            size_t size1 = l_Mesh->Indices.size();
            l_IndexBufferFile.write( (char *)&size1, sizeof( size1 ) );
            l_IndexBufferFile.write( (char *)&l_Mesh->Indices[0], l_Mesh->Indices.size() * sizeof( uint32_t ) );
            l_IndexBufferFile.flush();

            out << YAML::BeginMap;
            out << YAML::Key << "name" << YAML::Value << l_Mesh->Name;
            out << YAML::Key << "vertex_buffer" << YAML::Value << fmt::format( "meshes/{}/Vertices.dat", l_Mesh->Name );
            out << YAML::Key << "index_buffer" << YAML::Value << fmt::format( "meshes/{}/Indices.dat", l_Mesh->Name );
            out << YAML::Key << "material" << YAML::Value << l_MaterialPathMap[l_Mesh->Material.ID];
            out << YAML::Key << "bones" << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for( auto &l_Bone : l_Mesh->Bones )
                out << l_Bones[l_Bone.Name];
            out << YAML::EndSeq;
            out << YAML::EndMap;
        }
    }
    out << YAML::EndSeq;

    out << YAML::Key << "nodes" << YAML::Value << YAML::BeginSeq;
    {
        for( auto &l_Node : o_NodesList )
        {
            out << YAML::BeginMap;
            out << YAML::Key << "id" << YAML::Value << l_Node.ID;
            out << YAML::Key << "name" << YAML::Value << l_Node.Name;
            out << YAML::Key << "parent_id" << YAML::Value << l_Node.ParentID;

            out << YAML::Key << "transform" << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for( int i = 0; i < 4; i++ )
                for( int j = 0; j < 4; j++ )
                    out << l_Node.Transform[i][j];
            out << YAML::EndSeq;

            out << YAML::Key << "children_id" << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for( auto &i : l_Node.ChildrenID )
                out << i;
            out << YAML::EndSeq;

            out << YAML::Key << "meshes" << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for( auto &i : l_Node.Meshes )
                out << i->ID;
            out << YAML::EndSeq;

            out << YAML::EndMap;
        }
    }
    out << YAML::EndSeq;

    std::vector<std::shared_ptr<AnimationSequence>> Animations;
    if( ( l_Scene->mNumAnimations > 0 ) && ( l_Scene->mAnimations != nullptr ) )
        LoadAnimations( l_Scene, Animations, l_Nodes );

    out << YAML::Key << "animations" << YAML::Value << YAML::BeginSeq;
    {
        for( auto &l_Animation : Animations )
        {
            out << YAML::BeginMap;
            out << YAML::Key << "id" << YAML::Value << l_Animation->ID;
            out << YAML::Key << "name" << YAML::Value << l_Animation->Name;
            out << YAML::Key << "duration" << YAML::Value << l_Animation->Duration;
            out << YAML::Key << "tick_count" << YAML::Value << l_Animation->TickCount;
            out << YAML::Key << "ticks_per_second" << YAML::Value << l_Animation->TicksPerSecond;
            out << YAML::Key << "tracks" << YAML::Value << YAML::BeginSeq;
            for( auto &l_AnimationTrack : l_Animation->NodeAnimationTracks )
            {
                out << YAML::BeginMap;
                out << YAML::Key << "target_node_name" << YAML::Value << l_AnimationTrack.TargetNodeName;
                out << YAML::Key << "target_node_id" << YAML::Value << l_AnimationTrack.TargetNodeID;
                out << YAML::Key << "keyframes" << YAML::Value << YAML::BeginMap;
                out << YAML::Key << "translation" << YAML::Value << YAML::BeginSeq;
                for( auto &l_Translation : l_AnimationTrack.TranslationKeyFrames )
                    EmitKeyFrame( out, l_Translation );
                out << YAML::EndSeq;
                out << YAML::Key << "scaling" << YAML::Value << YAML::BeginSeq;
                for( auto &l_Scaling : l_AnimationTrack.ScalingKeyFrames )
                    EmitKeyFrame( out, l_Scaling );
                out << YAML::EndSeq;
                out << YAML::Key << "rotation" << YAML::Value << YAML::BeginSeq;
                for( auto &l_Rotation : l_AnimationTrack.RotationKeyFrames )
                    EmitKeyFrame( out, l_Rotation );
                out << YAML::EndSeq;
                out << YAML::EndMap;
                out << YAML::EndMap;
            }

            YAML::EndSeq;
            out << YAML::EndMap;
        }
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;

    std::ofstream fout( l_AssetConfigFile );
    fout << out.c_str();
}
