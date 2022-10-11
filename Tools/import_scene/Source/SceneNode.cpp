#include "SceneNode.h"

#include "Conversion.h"
#include "Core/Logging.h"
#include <fmt/core.h>

int32_t CountNodes( const aiNode *a_RootNode )
{
    if( a_RootNode->mNumChildren == 0 )
        return 1;

    int32_t l_Count = 1;
    for( uint32_t l_ChildIdx = 0; l_ChildIdx < a_RootNode->mNumChildren; l_ChildIdx++ )
        l_Count += CountNodes( a_RootNode->mChildren[l_ChildIdx] );
    return l_Count;
}

// Load the scene node hierarchy. This is a hierarchy of transform matrices each applying to a specific set of
// meshes. Our engine expects object parenting to be done through the transform. We therefore build a hierarchy
// of transformation matrices. Each mesh referred to in the node yields a new leaf node with the appropriate
// transformation. The names of the nodes are preserved.
int32_t LoadSceneNodes( int level, int32_t a_NodeID, const aiScene *a_SceneData, const aiNode *a_Node, NodeData &a_ParentNode,
                        const std::vector<std::shared_ptr<MeshData>> &l_Meshes, std::map<std::string, int32_t> &o_Nodes, std::vector<NodeData> &o_NodesList )
{
    auto &l_NewNode     = o_NodesList[a_NodeID];
    l_NewNode.ID        = a_NodeID;
    l_NewNode.Name      = a_Node->mName.C_Str();
    l_NewNode.Transform = to_mat4( a_Node->mTransformation );
    l_NewNode.ParentID  = a_ParentNode.ID;

    fmt::print( "{:{}}", "", level * 2 );
    fmt::print( "{} - ({} meshes)\n", l_NewNode.Name, a_Node->mNumMeshes );

    if( a_ParentNode.ID != -1 )
        a_ParentNode.ChildrenID.push_back( l_NewNode.ID );

    for( uint32_t l_MeshIdx = 0; l_MeshIdx < a_Node->mNumMeshes; l_MeshIdx++ )
        l_NewNode.Meshes.push_back( l_Meshes[a_Node->mMeshes[l_MeshIdx]] );

    o_Nodes[l_NewNode.Name] = l_NewNode.ID;

    if( a_Node->mNumChildren == 0 )
        return 1;

    int32_t l_SubtreeOffsetID = 1;
    for( uint32_t l_ChildIdx = 0; l_ChildIdx < a_Node->mNumChildren; l_ChildIdx++ )
    {
        int32_t l_SubtreeElementCount =
            LoadSceneNodes( level + 1, a_NodeID + l_SubtreeOffsetID, a_SceneData, a_Node->mChildren[l_ChildIdx], l_NewNode, l_Meshes, o_Nodes, o_NodesList );
        l_SubtreeOffsetID += l_SubtreeElementCount;
    }
    return l_SubtreeOffsetID;
}
