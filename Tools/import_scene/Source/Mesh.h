#pragma once

#include "Core/Math/Types.h"

#include "Developer/GraphicContext/Buffer.h"
#include "Developer/GraphicContext/GraphicsPipeline.h"

#include "Developer/Scene/VertexData.h"

#include "Material.h"

struct BoneData
{
    uint32_t ID;
    std::string Name;
    int32_t NodeID;
    std::vector<uint32_t> Vertices;
    std::vector<float> Weights;
    math::mat4 InverseBindMatrix;
};

struct AugmentedVertexData
{
    std::vector<uint32_t> Bones;
    std::vector<float> Weights;
};

struct MeshData
{
    uint32_t ID;
    std::string Name;
    LTSE::Graphics::PrimitiveTopology Primitive;
    std::vector<LTSE::Core::VertexData> Vertices;
    std::vector<uint32_t> Indices;
    std::vector<uint32_t> WireframeIndices;
    std::vector<BoneData> Bones;
    std::map<std::string, uint32_t> BoneMap;
    MaterialData Material;
    // Bounding Box
};

bool LoadMeshes( const aiScene *a_SceneData, std::vector<std::shared_ptr<MeshData>> &a_Meshes, std::vector<MaterialData> &a_Materials );
