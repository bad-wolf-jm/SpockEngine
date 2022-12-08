#pragma once

#include <filesystem>
#include <map>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "Core/Math/Types.h"

#include "SceneNode.h"

template <typename ValueType>
struct KeyFrame
{
    float     Tick;
    ValueType Value;
};

struct NodeAnimationTrack
{
    std::string                       TargetNodeName;
    int32_t                           TargetNodeID;
    std::vector<KeyFrame<math::vec3>> TranslationKeyFrames;
    std::vector<KeyFrame<math::vec3>> ScalingKeyFrames;
    std::vector<KeyFrame<math::quat>> RotationKeyFrames;
};

struct AnimationSequence
{
    uint32_t                        ID;
    std::string                     Name;
    float                           Duration;
    float                           TickCount;
    float                           TicksPerSecond;
    std::vector<NodeAnimationTrack> NodeAnimationTracks;
};

void LoadAnimations( const aiScene *a_SceneData, std::vector<std::shared_ptr<AnimationSequence>> &a_AnimationTracks,
                     std::map<std::string, int32_t> a_Nodes );
