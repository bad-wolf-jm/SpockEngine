#pragma once

#include <map>
#include <vector>


#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "Core/Math/Types.h"
#include "Mesh.h"

struct NodeData
{
    int32_t ID                      = -1;
    std::string Name                = "";
    int32_t ParentID                = -1;
    math::mat4 Transform            = math::mat4( 0.0f );
    std::vector<int32_t> ChildrenID = {};
    std::vector<std::shared_ptr<MeshData>> Meshes;
};

int32_t CountNodes( const aiNode *a_RootNode );

// Load the scene node hierarchy. This is a hierarchy of transform matrices each applying to a specific set of
// meshes. Our engine expects object parenting to be done through the transform. We therefore build a hierarchy
// of transformation matrices. Each mesh referred to in the node yields a new leaf node with the appropriate
// transformation. The names of the nodes are preserved.
int32_t LoadSceneNodes( int level, int32_t a_NodeID, const aiScene *a_SceneData, const aiNode *a_Node, NodeData &a_ParentNode,
                        const std::vector<std::shared_ptr<MeshData>> &l_Meshes, std::map<std::string, int32_t> &o_Nodes, std::vector<NodeData> &o_NodesList );
