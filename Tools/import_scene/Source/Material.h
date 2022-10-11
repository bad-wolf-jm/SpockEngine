#pragma once

#include <filesystem>
#include <map>
#include <vector>


#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "Core/Math/Types.h"
// #include "Graphic/Engine/GraphicContext/Texture2D.h"

enum TextureType
{
    DIFFUSE           = aiTextureType_DIFFUSE,
    SPECULAR          = aiTextureType_SPECULAR,
    AMBIENT           = aiTextureType_AMBIENT,
    EMISSIVE          = aiTextureType_EMISSIVE,
    OPACITY           = aiTextureType_OPACITY,
    NORMALS           = aiTextureType_NORMALS,
    HEIGHT            = aiTextureType_HEIGHT,
    SHININESS         = aiTextureType_SHININESS,
    DISPLACEMENT      = aiTextureType_DISPLACEMENT,
    LIGHTMAP          = aiTextureType_LIGHTMAP,
    REFLECTION        = aiTextureType_REFLECTION,
    BASE_COLOR        = aiTextureType_BASE_COLOR,
    NORMAL_CAMERA     = aiTextureType_NORMAL_CAMERA,
    EMISSION_COLOR    = aiTextureType_EMISSION_COLOR,
    METALNESS         = aiTextureType_METALNESS,
    DIFFUSE_ROUGHNESS = aiTextureType_DIFFUSE_ROUGHNESS,
    AMBIENT_OCCLUSION = aiTextureType_AMBIENT_OCCLUSION,
    UNKNOWN           = aiTextureType_UNKNOWN
};

struct MaterialData
{
    std::string Name;
    int32_t ID;
    bool IsWireframe;
    bool IsTwoSided;
    math::vec3 DiffuseColor;
    math::vec3 SpecularColor;
    math::vec3 AmbientColor;
    math::vec3 EmissiveColor;
    math::vec3 TransparentColor;
    math::vec3 ReflectiveColor;
    float Reflectivity;
    int BlendFunction;
    float Opacity;
    float Shininess;
    float ShininessStrength;
    float RefractionIndex;
    uint32_t Workflow;
    float MetallicFactor  = 1.0f;
    float RoughnessFactor = 1.0f;

    size_t TextureFlags;
    std::vector<std::string> TexturePaths;
    std::vector<uint32_t> TexCoords;
};

bool LoadMaterials( const aiScene *a_SceneData, std::filesystem::path a_AssetRoot, std::vector<MaterialData> &a_Materials );
