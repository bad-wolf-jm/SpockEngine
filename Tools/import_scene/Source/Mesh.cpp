#include "Mesh.h"

#include "Conversion.h"
#include "Core/Logging.h"


bool LoadMeshes( const aiScene *a_SceneData, std::vector<std::shared_ptr<MeshData>> &a_Meshes, std::vector<MaterialData> &a_Materials )
{
    SE::Logging::Info( "Loading meshes..." );
    a_Meshes.resize( a_SceneData->mNumMeshes );
    for( auto l_MeshIdx = 0; l_MeshIdx < a_SceneData->mNumMeshes; l_MeshIdx++ )
    {
        auto l_Mesh = a_SceneData->mMeshes[l_MeshIdx];

        std::shared_ptr<MeshData> l_MeshData = std::make_shared<MeshData>();
        l_MeshData->Name                     = l_Mesh->mName.C_Str();
        if( l_MeshData->Name.length() == 0 )
            l_MeshData->Name = fmt::format( "Unnamed mesh {}", l_MeshIdx );

        l_MeshData->ID        = l_MeshIdx;
        l_MeshData->Primitive = SE::Graphics::PrimitiveTopology::TRIANGLES;
        l_MeshData->Material  = a_Materials[l_Mesh->mMaterialIndex];

        l_MeshData->Vertices.resize( l_Mesh->mNumVertices );

        bool l_HasNormals            = ( l_Mesh->mNormals != nullptr );
        bool l_HasTangents           = ( l_Mesh->mTangents != nullptr );
        bool l_HasVertexColors       = ( l_Mesh->mColors[0] != nullptr );
        bool l_HasTextureCoordinates = ( l_Mesh->mTextureCoords[0] != nullptr );
        SE::Logging::Info( "  Mesh #{} with {} vertices (normals={}, tangents={}, colors={}, texcoords={})", l_MeshIdx + 1, l_Mesh->mNumVertices, l_HasNormals, l_HasTangents,
                             l_HasVertexColors, l_HasTextureCoordinates );

        // Process the vertex data: the data from Assimp's scene structure is converted into our internal
        // VertexData structure to be used by our shaders.
        for( auto l_VertexIdx = 0; l_VertexIdx < l_Mesh->mNumVertices; l_VertexIdx++ )
        {
            l_MeshData->Vertices[l_VertexIdx].Position = to_vec3( l_Mesh->mVertices[l_VertexIdx] );

            if( l_HasTextureCoordinates )
            {
                l_MeshData->Vertices[l_VertexIdx].TexCoords_0 = to_vec2( l_Mesh->mTextureCoords[0][l_VertexIdx] );
                l_MeshData->Vertices[l_VertexIdx].TexCoords_0.y = 1.0f - l_MeshData->Vertices[l_VertexIdx].TexCoords_0.y;
                if( l_Mesh->mTextureCoords[1] != nullptr )
                {
                    l_MeshData->Vertices[l_VertexIdx].TexCoords_1 = to_vec2( l_Mesh->mTextureCoords[1][l_VertexIdx] );
                    l_MeshData->Vertices[l_VertexIdx].TexCoords_1.y = 1.0f - l_MeshData->Vertices[l_VertexIdx].TexCoords_1.y;
                }
            }

            if( l_HasNormals )
                l_MeshData->Vertices[l_VertexIdx].Normal = to_vec3( l_Mesh->mNormals[l_VertexIdx] );
        }

        // Process the face data. Normally at this point each mesh should only contain one primitive type,
        // and that type should be LINES or TRIANGLES as per ASSIMP's doucmentation. Vertex indices are stored
        // as uint32_t
        for( auto l_FaceIdx = 0; l_FaceIdx < l_Mesh->mNumFaces; l_FaceIdx++ )
        {
            auto l_Face = l_Mesh->mFaces[l_FaceIdx];
            for( auto i = 0; i < l_Face.mNumIndices; i++ )
                l_MeshData->Indices.push_back( l_Face.mIndices[i] );
        }

        // Process bone data. Each bone can be referred to by name, and contains information on all
        // the vertices it influences.  This information should be used to produce an augmented vertex
        // structure which adds to each vertex an array of all those bones which influence it. The bones
        // themselves are stored in an array and we keep a mapping of bone name to bone index.

        std::vector<AugmentedVertexData> l_AugmentedVertices( l_Mesh->mNumVertices );

        for( auto l_BoneIdx = 0; l_BoneIdx < l_Mesh->mNumBones; l_BoneIdx++ )
        {
            auto l_Bone = l_Mesh->mBones[l_BoneIdx];
            SE::Logging::Info( "      Bone #{} - {}", l_BoneIdx, l_Bone->mName.C_Str() );
            BoneData l_BoneData;
            l_BoneData.Name              = l_Bone->mName.C_Str();
            l_BoneData.ID                = l_BoneIdx;
            l_BoneData.InverseBindMatrix = to_mat4( l_Bone->mOffsetMatrix );
            l_BoneData.Vertices.resize( l_Bone->mNumWeights );
            l_BoneData.Weights.resize( l_Bone->mNumWeights );
            for( auto l_WeightIdx = 0; l_WeightIdx < l_Bone->mNumWeights; l_WeightIdx++ )
            {
                l_BoneData.Vertices[l_WeightIdx] = l_Bone->mWeights[l_WeightIdx].mVertexId;
                l_BoneData.Weights[l_WeightIdx]  = l_Bone->mWeights[l_WeightIdx].mWeight;

                // Add the bone to the appropriate vertex data. This is the information that will be used by
                // shaders to animate the model
                l_AugmentedVertices[l_BoneData.Vertices[l_WeightIdx]].Bones.push_back( l_BoneIdx );
                l_AugmentedVertices[l_BoneData.Vertices[l_WeightIdx]].Weights.push_back( l_BoneData.Weights[l_WeightIdx] );
            }
            l_MeshData->Bones.push_back( l_BoneData );
            l_MeshData->BoneMap[l_BoneData.Name] = l_BoneData.ID;
        }

        // Fill the vertex data bone and weight information. We keep only the 4 most significant weights.
        auto i = 0;
        for( auto &l_Vertex : l_MeshData->Vertices )
        {
            auto &l_BoneInfo = l_AugmentedVertices[i];
            auto l_BoneCount = l_BoneInfo.Bones.size();

            l_Vertex.Bones.x = ( l_BoneCount > 0 ) ? l_BoneInfo.Bones[0] : 0.0f;
            l_Vertex.Bones.y = ( l_BoneCount > 1 ) ? l_BoneInfo.Bones[1] : 0.0f;
            l_Vertex.Bones.z = ( l_BoneCount > 2 ) ? l_BoneInfo.Bones[2] : 0.0f;
            l_Vertex.Bones.w = ( l_BoneCount > 3 ) ? l_BoneInfo.Bones[3] : 0.0f;

            l_Vertex.Weights.x = ( l_BoneCount > 0 ) ? l_BoneInfo.Weights[0] : 0.0f;
            l_Vertex.Weights.y = ( l_BoneCount > 1 ) ? l_BoneInfo.Weights[1] : 0.0f;
            l_Vertex.Weights.z = ( l_BoneCount > 2 ) ? l_BoneInfo.Weights[2] : 0.0f;
            l_Vertex.Weights.w = ( l_BoneCount > 3 ) ? l_BoneInfo.Weights[3] : 0.0f;

            // If there are more than 4 bones, the weights of the remaining bones are added and redistributed
            // evenly among the first 4 bones.
            if( l_BoneCount > 4 )
            {
                float l_RemainingWeight = 1.0f - ( l_Vertex.Weights.x, l_Vertex.Weights.y, l_Vertex.Weights.z, l_Vertex.Weights.w );

                if( l_RemainingWeight > 0.0f )
                {
                    l_Vertex.Weights.x += l_RemainingWeight / 4.0f;
                    l_Vertex.Weights.y += l_RemainingWeight / 4.0f;
                    l_Vertex.Weights.z += l_RemainingWeight / 4.0f;
                    l_Vertex.Weights.w += l_RemainingWeight / 4.0f;
                }
            }

            i++;
        }

        a_Meshes[l_MeshIdx] = l_MeshData;
    }
    return true;
}
