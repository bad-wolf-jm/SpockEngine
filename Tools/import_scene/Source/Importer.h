#pragma once

#include <filesystem>
#include <string>
#include <vector>


#include "Animation.h"
#include "Material.h"
#include "Mesh.h"
#include "SceneNode.h"


namespace fs = std::filesystem;

struct TextureData
{
    std::string Name;
    std::string Path;
};

struct CameraData
{
    std::string Name;
    bool x;
};

struct LightData
{
    std::string Name;
    bool x;
};

struct AssetData
{
    std::shared_ptr<NodeData> RootNode;
    std::vector<NodeData> Nodes;
    std::vector<std::shared_ptr<AnimationSequence>> Animations;
};

void ReadAssetFile( const fs::path &a_FilePath, const fs::path &a_YamlFilePath );
