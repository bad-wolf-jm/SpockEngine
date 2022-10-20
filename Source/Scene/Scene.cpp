#include <filesystem>
#include <fmt/core.h>
#include <future>
#include <gli/gli.hpp>
#include <iostream>
#include <queue>
#include <stack>
#include <unordered_map>

#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

#include "Core/Vulkan/VkPipeline.h"

#include "Components.h"
#include "Scene.h"
#include "VertexTransform.h"

#include "Renderer/MeshRenderer.h"

#include "Core/GraphicContext//Texture2D.h"
#include "Scripting/ScriptComponent.h"

#include "Scene/Components/VisualHelpers.h"
// #include "LidarSensorModel/AcquisitionContext/AcquisitionContext.h"
// #include "LidarSensorModel/EnvironmentSampler.h"

namespace LTSE::Core
{

    namespace fs = std::filesystem;
    using namespace LTSE::Graphics;
    using namespace LTSE::Cuda;
    // using namespace LTSE::SensorModel;
    using namespace LTSE::Core::EntityComponentSystem;
    using namespace LTSE::Core::EntityComponentSystem::Components;

    Scene::Scene( GraphicContext &a_GraphicContext, Ref<LTSE::Core::UIContext> a_UI )
        : mGraphicContext{ a_GraphicContext }
        , m_UI{ a_UI }
    {

        mSceneScripting = New<ScriptingEngine>();
        mMaterialSystem = New<MaterialSystem>( a_GraphicContext );

        DefaultCamera                 = m_Registry.CreateEntity( "DefaultCamera" );
        auto &l_CameraComponent       = DefaultCamera.Add<CameraComponent>();
        l_CameraComponent.Position    = math::vec3{ 0.0f, 0.0f, 0.0f };
        l_CameraComponent.Pitch       = 0.0f;
        l_CameraComponent.Yaw         = 0.0f;
        l_CameraComponent.Roll        = 0.0f;
        l_CameraComponent.Near        = 0.001;
        l_CameraComponent.Far         = 1000.0f;
        l_CameraComponent.FieldOfView = 90.0f;
        l_CameraComponent.AspectRatio = 16.0f / 9.0f;

        CurrentCamera = DefaultCamera;

        Environment = m_Registry.CreateEntity( "Environment" );
        Environment.Add<AmbientLightingComponent>();
        Environment.Add<BackgroundComponent>();

        Root = m_Registry.CreateEntityWithRelationship( "WorldRoot" );
        Root.Add<LocalTransformComponent>();

        InitializeRayTracing();
        ConnectSignalHandlers();
    }

    template <typename _Component>
    static void CopyComponent( Entity &aSource, Entity &aDestination )
    {
        if( ( aSource.Has<_Component>() ) ) aDestination.AddOrReplace<_Component>( aSource.Get<_Component>() );
    }

    Scene::Scene( Ref<Scene> aSource )
    {
        mGraphicContext = aSource->mGraphicContext;
        m_UI            = aSource->m_UI;

        mSceneScripting = aSource->mSceneScripting;
        mMaterialSystem = aSource->mMaterialSystem;

        InitializeRayTracing();

        std::unordered_map<UUIDv4::UUID, Entity> lSourceEntities{};
        std::unordered_map<UUIDv4::UUID, Entity> lClonedEntities{};

        aSource->ForEach<sUUID>(
            [&]( auto aEntity, auto &aUUID )
            {
                auto lClonedEntity = CreateEntity();

                CopyComponent<sUUID>( aEntity, lClonedEntity );
                CopyComponent<sTag>( aEntity, lClonedEntity );

                lSourceEntities[aUUID.mValue] = aEntity;
                lClonedEntities[aUUID.mValue] = lClonedEntity;
            } );

        // Copy simple components
        for( auto &[lUUID, lEntity] : lSourceEntities )
        {
            auto lClonedEntity = lClonedEntities[lUUID];

            CopyComponent<LocalTransformComponent>( lEntity, lClonedEntity );
            CopyComponent<TransformMatrixComponent>( lEntity, lClonedEntity );
            CopyComponent<AnimatedTransformComponent>( lEntity, lClonedEntity );

            CopyComponent<StaticMeshComponent>( lEntity, lClonedEntity );
            CopyComponent<WireframeComponent>( lEntity, lClonedEntity );
            CopyComponent<WireframeMeshComponent>( lEntity, lClonedEntity );
            CopyComponent<BoundingBoxComponent>( lEntity, lClonedEntity );

            CopyComponent<ParticleSystemComponent>( lEntity, lClonedEntity );
            CopyComponent<ParticleShaderComponent>( lEntity, lClonedEntity );

            CopyComponent<RayTracingTargetComponent>( lEntity, lClonedEntity );

            CopyComponent<MaterialComponent>( lEntity, lClonedEntity );
            CopyComponent<RendererComponent>( lEntity, lClonedEntity );
            CopyComponent<MaterialShaderComponent>( lEntity, lClonedEntity );

            CopyComponent<BackgroundComponent>( lEntity, lClonedEntity );

            CopyComponent<AmbientLightingComponent>( lEntity, lClonedEntity );
            CopyComponent<DirectionalLightComponent>( lEntity, lClonedEntity );
            CopyComponent<PointLightComponent>( lEntity, lClonedEntity );
            CopyComponent<SpotlightComponent>( lEntity, lClonedEntity );
            CopyComponent<LightComponent>( lEntity, lClonedEntity );

            CopyComponent<sBehaviourComponent>( lEntity, lClonedEntity );
            CopyComponent<sLuaScriptComponent>( lEntity, lClonedEntity );

            CopyComponent<PointLightHelperComponent>( lEntity, lClonedEntity );
            CopyComponent<DirectionalLightHelperComponent>( lEntity, lClonedEntity );
            CopyComponent<FieldOfViewHelperComponent>( lEntity, lClonedEntity );
            CopyComponent<CameraHelperComponent>( lEntity, lClonedEntity );

            // Move somewhere else
            // CopyComponent<EnvironmentSampler::sCreateInfo>( lEntity, lClonedEntity );
            // CopyComponent<AcquisitionSpecification>( lEntity, lClonedEntity );

            CopyComponent<SkeletonComponent>( lEntity, lClonedEntity );
            if( lClonedEntity.Has<SkeletonComponent>() )
            {
                auto &lSkeletonComponent = lClonedEntity.Get<SkeletonComponent>();
                for( uint32_t i = 0; i < lSkeletonComponent.BoneCount; i++ )
                    lSkeletonComponent.Bones[i] = lClonedEntities[lSkeletonComponent.Bones[i].Get<sUUID>().mValue];
            }

            CopyComponent<AnimationComponent>( lEntity, lClonedEntity );
            if( lClonedEntity.Has<AnimationComponent>() )
            {
                auto &lAnimationComponent = lClonedEntity.Get<AnimationComponent>();
                for( auto &lChannel : lAnimationComponent.mChannels )
                    lChannel.mTargetNode = lClonedEntities[lChannel.mTargetNode.Get<sUUID>().mValue];
            }

            CopyComponent<AnimationChooser>( lEntity, lClonedEntity );
            if( lClonedEntity.Has<AnimationChooser>() )
            {
                auto &lAnimationChooser = lClonedEntity.Get<AnimationChooser>();
                for( uint32_t i = 0; i < lAnimationChooser.Animations.size(); i++ )
                    lAnimationChooser.Animations[i] = lClonedEntities[lAnimationChooser.Animations[i].Get<sUUID>().mValue];
            }
        }

        // Copy the hierarchy
        for( auto &[lUUID, lEntity] : lSourceEntities )
        {
            auto &lDestEntity = lClonedEntities[lUUID];

            if( lEntity.Has<sRelationshipComponent>() )
            {
                auto &lSourceParentEntity = lEntity.Get<sRelationshipComponent>().mParent;

                if( lSourceParentEntity )
                {
                    auto &lDestParentEntity = lClonedEntities[lSourceParentEntity.Get<sUUID>().mValue];

                    m_Registry.SetParent( lDestEntity, lDestParentEntity );
                }
            }
        }

        Root = lClonedEntities[aSource->Root.Get<sUUID>().mValue];

        Environment = lClonedEntities[aSource->Environment.Get<sUUID>().mValue];

        DefaultCamera = lClonedEntities[aSource->DefaultCamera.Get<sUUID>().mValue];
        CurrentCamera = lClonedEntities[aSource->CurrentCamera.Get<sUUID>().mValue];

        // Copy a reference to the main vertex buffer and its CUDA handle
        mVertexBuffer             = aSource->mVertexBuffer;
        mVertexBufferMemoryHandle = aSource->mVertexBufferMemoryHandle;

        // Copy a reference to the main index buffer and its CUDA handle
        mIndexBuffer             = aSource->mIndexBuffer;
        mIndexBufferMemoryHandle = aSource->mIndexBufferMemoryHandle;

        // Create the transformed vertex buffer and its CUDA handle
        mTransformedVertexBuffer = New<Buffer>(
            mGraphicContext, eBufferBindType::VERTEX_BUFFER, false, true, true, true, mVertexBuffer->SizeAs<uint8_t>() );
        mTransformedVertexBufferMemoryHandle =
            Cuda::GPUExternalMemory( *mTransformedVertexBuffer, mTransformedVertexBuffer->SizeAs<uint8_t>() );

        uint32_t lTransformCount = 0;
        aSource->ForEach<LocalTransformComponent>( [&]( auto aEntity, auto &aUUID ) { lTransformCount++; } );

        uint32_t lStaticMeshCount = 0;
        aSource->ForEach<StaticMeshComponent>( [&]( auto aEntity, auto &aUUID ) { lStaticMeshCount++; } );

        uint32_t lJointMatrixCount = 0;
        uint32_t lJointOffsetCount = 0;
        ForEach<SkeletonComponent>(
            [&]( auto l_ElementToProcess, auto &s )
            {
                lJointMatrixCount += s.JointMatrices.size();
                lJointOffsetCount += 1;
            } );

        mTransforms      = GPUMemory::Create<math::mat4>( lTransformCount );
        mVertexOffsets   = GPUMemory::Create<uint32_t>( lStaticMeshCount );
        mVertexCounts    = GPUMemory::Create<uint32_t>( lStaticMeshCount );
        mJointTransforms = GPUMemory::Create<math::mat4>( lJointMatrixCount );
        mJointOffsets    = GPUMemory::Create<uint32_t>( lJointOffsetCount );

        ConnectSignalHandlers();
        mIsClone = true;
    }

    Scene::~Scene()
    {
        if( !mIsClone )
        {
            mVertexBufferMemoryHandle.Dispose();
            mIndexBufferMemoryHandle.Dispose();
        }

        mTransformedVertexBufferMemoryHandle.Dispose();
        mTransforms.Dispose();
        mVertexOffsets.Dispose();
        mVertexCounts.Dispose();
    }

    void Scene::UpdateParent( Entity const &aEntity, sRelationshipComponent const &aComponent )
    {
        auto &lParent = aComponent.mParent;

        if( !( aEntity.Has<LocalTransformComponent>() ) ) return;

        auto &lLocalTransform = aEntity.Get<LocalTransformComponent>();

        if( lParent && lParent.Has<TransformMatrixComponent>() )
        {
            aEntity.AddOrReplace<TransformMatrixComponent>(
                lParent.Get<TransformMatrixComponent>().Matrix * lLocalTransform.mMatrix );

            UpdateTransformMatrix( aEntity, aEntity.Get<TransformMatrixComponent>() );
        }
    }

    void Scene::UpdateLocalTransform( Entity const &aEntity, LocalTransformComponent const &aComponent )
    {
        auto &lParent = aEntity.Get<sRelationshipComponent>().mParent;

        if( lParent && lParent.Has<TransformMatrixComponent>() )
            aEntity.AddOrReplace<TransformMatrixComponent>(
                lParent.Get<TransformMatrixComponent>().Matrix * aComponent.mMatrix );
        else
            aEntity.AddOrReplace<TransformMatrixComponent>( aComponent.mMatrix );

        UpdateTransformMatrix( aEntity, aEntity.Get<TransformMatrixComponent>() );
    }

    void Scene::UpdateTransformMatrix( Entity const &aEntity, TransformMatrixComponent const &aComponent )
    {
        if( !aEntity.Has<sRelationshipComponent>() ) return;

        for( auto lChild : aEntity.Get<sRelationshipComponent>().mChildren )
        {
            if( lChild.Has<LocalTransformComponent>() )
            {
                lChild.AddOrReplace<TransformMatrixComponent>(
                    aComponent.Matrix * lChild.Get<LocalTransformComponent>().mMatrix );

                UpdateTransformMatrix( lChild, lChild.Get<TransformMatrixComponent>() );
            }
        }
    }

    void Scene::ConnectSignalHandlers()
    {
        using namespace std::placeholders;

        m_Registry.OnComponentAdded<sRelationshipComponent>( std::bind( &Scene::UpdateParent, this, _1, _2 ) );
        m_Registry.OnComponentUpdated<sRelationshipComponent>( std::bind( &Scene::UpdateParent, this, _1, _2 ) );

        m_Registry.OnComponentAdded<LocalTransformComponent>( std::bind( &Scene::UpdateLocalTransform, this, _1, _2 ) );
        m_Registry.OnComponentUpdated<LocalTransformComponent>( std::bind( &Scene::UpdateLocalTransform, this, _1, _2 ) );

        m_Registry.OnComponentAdded<TransformMatrixComponent>( std::bind( &Scene::UpdateTransformMatrix, this, _1, _2 ) );
        m_Registry.OnComponentUpdated<TransformMatrixComponent>( std::bind( &Scene::UpdateTransformMatrix, this, _1, _2 ) );
    }

    math::mat4 Scene::GetView()
    {
        math::mat4 l_CameraView( 1.0f );
        if( CurrentCamera.Has<CameraComponent>() )
        {
            auto &l_Component = CurrentCamera.Get<CameraComponent>();

            math::mat4 l_Rx = math::Rotation( l_Component.Pitch, math::vec3( 1.0f, 0.0f, 0.0f ) );
            math::mat4 l_Ry = math::Rotation( l_Component.Yaw, math::vec3( 0.0f, 1.0f, 0.0f ) );
            math::mat4 l_Rz = math::Rotation( -l_Component.Roll, math::vec3( 0.0f, 0.0f, 1.0f ) );

            l_CameraView = math::Inverse( math::Translate( l_Rx * l_Ry * l_Rz, l_Component.Position ) );
        }

        return l_CameraView;
    }

    math::mat4 Scene::GetProjection()
    {
        math::mat4 l_CameraProjection( 1.0f );
        if( CurrentCamera.Has<CameraComponent>() )
        {
            auto &l_Component  = CurrentCamera.Get<CameraComponent>();
            l_CameraProjection = math::Perspective(
                math::radians( l_Component.FieldOfView ), l_Component.AspectRatio, l_Component.Near, l_Component.Far );
            l_CameraProjection[1][1] *= -1.0f;
        }
        return l_CameraProjection;
    }

    math::vec3 Scene::GetCameraPosition()
    {
        math::vec3 l_CameraPosition( 0.0f );
        if( CurrentCamera.Has<CameraComponent>() )
        {
            auto &l_Component = CurrentCamera.Get<CameraComponent>();
            l_CameraPosition  = l_Component.Position;
        }
        return l_CameraPosition;
    }

    Scene::Element Scene::Create( std::string a_Name, Element a_Parent )
    {
        return m_Registry.CreateEntity( a_Parent, a_Name );
    }

    Scene::Element Scene::CreateEntity() { return m_Registry.CreateEntity(); }

    Scene::Element Scene::CreateEntity( std::string a_Name ) { return m_Registry.CreateEntity( a_Name ); }

    void Scene::ClearScene()
    {
        m_Registry.Clear();
        ConnectSignalHandlers();

        mMaterialSystem->Clear();

        DefaultCamera                 = m_Registry.CreateEntity( "DefaultCamera" );
        auto &l_CameraComponent       = DefaultCamera.Add<CameraComponent>();
        l_CameraComponent.Position    = math::vec3{ 0.0f, 0.0f, 0.0f };
        l_CameraComponent.Pitch       = 0.0f;
        l_CameraComponent.Yaw         = 0.0f;
        l_CameraComponent.Roll        = 0.0f;
        l_CameraComponent.Near        = 0.001;
        l_CameraComponent.Far         = 1000.0f;
        l_CameraComponent.FieldOfView = 90.0f;
        l_CameraComponent.AspectRatio = 16.0f / 9.0f;

        CurrentCamera = DefaultCamera;

        Environment = m_Registry.CreateEntity( "Environment" );
        Environment.Add<AmbientLightingComponent>();
        Environment.Add<BackgroundComponent>();

        Root = m_Registry.CreateEntityWithRelationship( "WorldRoot" );
        Root.Add<LocalTransformComponent>();
    }

    Scene::Element Scene::LoadModel( Ref<sImportedModel> aModelData, math::mat4 aTransform )
    {
        return LoadModel( aModelData, aTransform, "MODEL" );
    }

    Scene::Element Scene::LoadModel( Ref<sImportedModel> aModelData, math::mat4 aTransform, std::string a_Name )
    {
        auto l_AssetEntity = m_Registry.CreateEntity( Root, a_Name );
        l_AssetEntity.Add<LocalTransformComponent>( aTransform );

        std::vector<uint32_t> lTextureIds = {};
        for( auto &lTexture : aModelData->mTextures )
        {
            lTextureIds.push_back( mMaterialSystem->CreateTexture( lTexture.mTexture, lTexture.mSampler ) );
        }

        std::vector<uint32_t>                lMaterialIds        = {};
        std::vector<MaterialShaderComponent> lMaterialCreateInfo = {};
        for( auto &lMaterial : aModelData->mMaterials )
        {
            auto &lNewMaterial = mMaterialSystem->CreateMaterial();
            lNewMaterial.mName = lMaterial.mName;

            lNewMaterial.mType = eMaterialType::Opaque;

            lNewMaterial.mLineWidth      = 1.0f;
            lNewMaterial.mIsTwoSided     = lMaterial.mConstants.mIsTwoSided;
            lNewMaterial.mUseAlphaMask   = false;
            lNewMaterial.mAlphaThreshold = 0.5;

            auto lGetTexID = [&]( uint32_t aID, uint32_t aDefault )
            { return aID == std::numeric_limits<uint32_t>::max() ? aDefault : lTextureIds[aID]; };

            lNewMaterial.mBaseColorFactor             = lMaterial.mConstants.mBaseColorFactor;
            lNewMaterial.mBaseColorTexture.mTextureID = lGetTexID( lMaterial.mTextures.mBaseColorTexture.TextureID, 1 );
            lNewMaterial.mBaseColorTexture.mUVChannel = lMaterial.mTextures.mBaseColorTexture.UVChannel;

            lNewMaterial.mEmissiveFactor             = lMaterial.mConstants.mEmissiveFactor;
            lNewMaterial.mEmissiveTexture.mTextureID = lGetTexID( lMaterial.mTextures.mEmissiveTexture.TextureID, 0 );
            lNewMaterial.mEmissiveTexture.mUVChannel = lMaterial.mTextures.mEmissiveTexture.UVChannel;

            lNewMaterial.mRoughnessFactor = lMaterial.mConstants.mRoughnessFactor;
            lNewMaterial.mMetallicFactor  = lMaterial.mConstants.mMetallicFactor;
            lNewMaterial.mMetalRoughTexture.mTextureID =
                lGetTexID( lMaterial.mTextures.mMetallicRoughnessTexture.TextureID, 0 );
            lNewMaterial.mMetalRoughTexture.mUVChannel = lMaterial.mTextures.mMetallicRoughnessTexture.UVChannel;

            lNewMaterial.mOcclusionStrength           = 0.0f;
            lNewMaterial.mOcclusionTexture.mTextureID = lGetTexID( lMaterial.mTextures.mOcclusionTexture.TextureID, 1 );
            lNewMaterial.mOcclusionTexture.mUVChannel = lMaterial.mTextures.mOcclusionTexture.UVChannel;

            lNewMaterial.mNormalsTexture.mTextureID = lGetTexID( lMaterial.mTextures.mNormalTexture.TextureID, 0 );
            lNewMaterial.mNormalsTexture.mUVChannel = lMaterial.mTextures.mNormalTexture.UVChannel;
            lMaterialIds.push_back( lNewMaterial.mID );

            MaterialShaderComponent lMaterialShader{};
            lMaterialShader.Type              = MaterialType::Opaque;
            lMaterialShader.IsTwoSided        = lMaterial.mConstants.mIsTwoSided;
            lMaterialShader.UseAlphaMask      = true;
            lMaterialShader.LineWidth         = 1.0f;
            lMaterialShader.AlphaMaskTheshold = 0.5;

            lMaterialCreateInfo.push_back( lMaterialShader );
        }

        std::vector<Element>    lMeshes           = {};
        uint32_t                lVertexBufferSize = 0;
        uint32_t                lIndexBufferSize  = 0;
        std::vector<VertexData> lVertexData       = {};
        std::vector<uint32_t>   lIndexData        = {};
        for( auto &lMesh : aModelData->mMeshes )
        {
            StaticMeshComponent l_MeshComponent{};
            l_MeshComponent.Name      = lMesh.mName;
            l_MeshComponent.Primitive = lMesh.mPrimitive;

            std::vector<VertexData> lVertices( lMesh.mPositions.size() );
            for( uint32_t i = 0; i < lMesh.mPositions.size(); i++ )
            {
                lVertices[i].Position    = math::vec4( lMesh.mPositions[i], 1.0f );
                lVertices[i].Normal      = lMesh.mNormals[i];
                lVertices[i].TexCoords_0 = lMesh.mUV0[i];
                lVertices[i].TexCoords_1 = lMesh.mUV1[i];
                lVertices[i].Bones       = lMesh.mJoints[i];
                lVertices[i].Weights     = lMesh.mWeights[i];
            }

            l_MeshComponent.Vertices =
                New<Buffer>( mGraphicContext, lVertices, eBufferBindType::VERTEX_BUFFER, false, true, true, true );
            l_MeshComponent.Indices =
                New<Buffer>( mGraphicContext, lMesh.mIndices, eBufferBindType::INDEX_BUFFER, false, true, true, true );
            l_MeshComponent.mVertexOffset = lVertexData.size();
            l_MeshComponent.mVertexCount  = lVertices.size();
            l_MeshComponent.mIndexOffset  = lIndexData.size();
            l_MeshComponent.mIndexCount   = lMesh.mIndices.size();

            auto l_MeshEntity = Create( lMesh.mName, l_AssetEntity );
            l_MeshEntity.Add<StaticMeshComponent>( l_MeshComponent );
            l_MeshEntity.Add<MaterialComponent>( lMaterialIds[lMesh.mMaterialID] );
            l_MeshEntity.Add<MaterialShaderComponent>( lMaterialCreateInfo[lMesh.mMaterialID] );
            l_MeshEntity.Add<LocalTransformComponent>( math::mat4( 1.0f ) );

            lVertexData.insert( lVertexData.end(), lVertices.begin(), lVertices.end() );
            lIndexData.insert( lIndexData.end(), lMesh.mIndices.begin(), lMesh.mIndices.end() );
            lMeshes.push_back( l_MeshEntity );
        }

        mVertexBuffer = New<Buffer>( mGraphicContext, lVertexData, eBufferBindType::VERTEX_BUFFER, false, true, true, true );
        mVertexBufferMemoryHandle.Dispose();
        mVertexBufferMemoryHandle = Cuda::GPUExternalMemory( *mVertexBuffer, mVertexBuffer->SizeAs<uint8_t>() );

        mIndexBuffer = New<Buffer>( mGraphicContext, lIndexData, eBufferBindType::INDEX_BUFFER, false, true, true, true );
        mIndexBufferMemoryHandle.Dispose();
        mIndexBufferMemoryHandle = Cuda::GPUExternalMemory( *mIndexBuffer, mIndexBuffer->SizeAs<uint8_t>() );

        mTransformedVertexBuffer = New<Buffer>( mGraphicContext, eBufferBindType::VERTEX_BUFFER, false, true, true, true,
            lVertexData.size() * sizeof( VertexData ) );
        mTransformedVertexBufferMemoryHandle.Dispose();
        mTransformedVertexBufferMemoryHandle =
            Cuda::GPUExternalMemory( *mTransformedVertexBuffer, mTransformedVertexBuffer->SizeAs<uint8_t>() );

        mTransforms    = GPUMemory::Create<math::mat4>( static_cast<uint32_t>( lMeshes.size() ) );
        mVertexOffsets = GPUMemory::Create<uint32_t>( static_cast<uint32_t>( lMeshes.size() ) );
        mVertexCounts  = GPUMemory::Create<uint32_t>( static_cast<uint32_t>( lMeshes.size() ) );

        std::vector<Element> lNodes = {};
        for( auto &lNode : aModelData->mNodes )
        {
            auto l_NodeEntity = m_Registry.CreateEntityWithRelationship( lNode.mName );
            l_NodeEntity.Add<LocalTransformComponent>( lNode.mTransform );

            lNodes.push_back( l_NodeEntity );
        }

        int32_t l_Index = 0;
        for( auto &l_NodeEntity : lNodes )
        {
            if( aModelData->mNodes[l_Index].mParentID == std::numeric_limits<uint32_t>::max() )
                m_Registry.SetParent( l_NodeEntity, l_AssetEntity );
            else
                m_Registry.SetParent( l_NodeEntity, lNodes[aModelData->mNodes[l_Index].mParentID] );
            l_Index++;
        }

        for( uint32_t lNodeID = 0; lNodeID < aModelData->mNodes.size(); lNodeID++ )
        {
            auto &lNode = aModelData->mNodes[lNodeID];

            if( lNode.mMeshes.size() == 0 ) continue;

            for( uint32_t lNodeMeshID = 0; lNodeMeshID < lNode.mMeshes.size(); lNodeMeshID++ )
                m_Registry.SetParent( lMeshes[lNodeMeshID], lNodes[lNodeID] );

            if( lNode.mSkinID != std::numeric_limits<uint32_t>::max() )
            {
                auto &lSkin = aModelData->mSkins[lNode.mSkinID];

                SkeletonComponent lNodeSkeleton{};

                std::vector<Element> lJointNodes = {};
                for( uint32_t i = 0; i < lSkin.mJointNodeID.size(); i++ )
                {
                    lNodeSkeleton.Bones.push_back( lNodes[lSkin.mJointNodeID[i]] );
                    lNodeSkeleton.InverseBindMatrices.push_back( lSkin.mInverseBindMatrices[i] );
                    lNodeSkeleton.JointMatrices.push_back( math::mat4( 0.0f ) );
                }

                lNodeSkeleton.BoneCount = lNodeSkeleton.Bones.size();
                for( uint32_t i = 0; i < lNode.mMeshes.size(); i++ )
                {
                    auto &lMesh = lMeshes[lNode.mMeshes[i]];
                    lMesh.Add<SkeletonComponent>( lNodeSkeleton );
                }
            }
        }

        uint32_t lJointMatrixCount = 0;
        uint32_t lJointOffsetCount = 0;
        ForEach<SkeletonComponent>(
            [&]( auto l_ElementToProcess, auto &s )
            {
                lJointMatrixCount += s.JointMatrices.size();
                lJointOffsetCount += 1;
            } );

        mJointTransforms = GPUMemory::Create<math::mat4>( lJointMatrixCount );
        mJointOffsets    = GPUMemory::Create<uint32_t>( lJointOffsetCount );

        if( aModelData->mAnimations.size() > 0 ) l_AssetEntity.Add<AnimationChooser>();

        for( auto &lAnimation : aModelData->mAnimations )
        {
            auto &l_AnimationChooser = l_AssetEntity.Get<AnimationChooser>();

            auto  l_AnimationEntity    = m_Registry.CreateEntity( l_AssetEntity, lAnimation.mName );
            auto &l_AnimationComponent = l_AnimationEntity.Add<AnimationComponent>();

            l_AnimationChooser.Animations.push_back( l_AnimationEntity );
            l_AnimationComponent.Duration = lAnimation.mEnd - lAnimation.mStart;

            for( uint32_t lAnimationChannelIndex = 0; lAnimationChannelIndex < lAnimation.mChannels.size();
                 lAnimationChannelIndex++ )
            {
                AnimationChannel lAnimationChannel{};
                lAnimationChannel.mChannelID = lAnimation.mChannels[lAnimationChannelIndex].mComponent;
                lAnimationChannel.mInterpolation =
                    lAnimation.mSamplers[lAnimation.mChannels[lAnimationChannelIndex].mSamplerIndex];
                lAnimationChannel.mTargetNode = lNodes[lAnimation.mChannels[lAnimationChannelIndex].mNodeID];
                lAnimationChannel.mTargetNode.TryAdd<AnimatedTransformComponent>();

                l_AnimationComponent.mChannels.push_back( lAnimationChannel );
            }
        }

        return l_AssetEntity;
    }

    void Scene::MarkAsRayTracingTarget( Scene::Element a_Element )
    {
        if( !a_Element.Has<StaticMeshComponent>() ) return;

        if( a_Element.Has<RayTracingTargetComponent>() ) return;

        auto &lRTComponent = a_Element.Add<RayTracingTargetComponent>();
    }

    void Scene::AttachScript( Element aElement, fs::path aScriptPath )
    {
        auto &lNewScriptComponent = aElement.Add<sLuaScriptComponent>( mSceneScripting, aScriptPath );
        lNewScriptComponent.Initialize( aElement );
    }

    void Scene::BeginScenario()
    {
        if( mState != eSceneState::EDITING ) return;

        // Initialize native scripts
        ForEach<sBehaviourComponent>(
            [=]( auto l_Entity, auto &l_Component )
            {
                if( !l_Component.ControllerInstance )
                {
                    l_Component.ControllerInstance = l_Component.InstantiateController();
                    l_Component.ControllerInstance->Initialize( m_Registry.WrapEntity( l_Entity ) );
                    l_Component.ControllerInstance->OnCreate();
                }
            } );

        // Initialize Lua scripts
        ForEach<sLuaScriptComponent>( [=]( auto l_Entity, auto &l_Component ) { l_Component.OnCreate(); } );

        mState = eSceneState::RUNNING;
    }

    void Scene::EndScenario()
    {
        if( mState != eSceneState::RUNNING ) return;

        // Destroy scripts
        ForEach<sBehaviourComponent>(
            [=]( auto l_Entity, auto &l_Component )
            {
                if( l_Component.ControllerInstance )
                {
                    l_Component.ControllerInstance->OnDestroy();
                    l_Component.DestroyController( &l_Component );
                }
            } );

        // Destroy Lua scripts
        ForEach<sLuaScriptComponent>( [=]( auto l_Entity, auto &l_Component ) { l_Component.OnDestroy(); } );

        mState = eSceneState::EDITING;
    }

    void Scene::Update( Timestep ts )
    {
        LTSE_PROFILE_FUNCTION();

        // Run scripts if the scene is in RUNNING mode.  The native scripts are run first, followed by the Lua scripts.
        if( mState == eSceneState::RUNNING )
        {
            ForEach<sBehaviourComponent>(
                [=]( auto l_Entity, auto &l_Component )
                {
                    if( l_Component.ControllerInstance ) l_Component.ControllerInstance->OnUpdate( ts );
                } );

            ForEach<sLuaScriptComponent>( [=]( auto l_Entity, auto &l_Component ) { l_Component.OnUpdate( ts ); } );

            // Update animations
            ForEach<AnimationChooser>(
                [=]( auto l_Entity, auto &l_Component )
                {
                    auto &lAnimation = l_Component.Animations[0].Get<AnimationComponent>();
                    lAnimation.CurrentTime += ts;
                    if( lAnimation.CurrentTime > lAnimation.Duration ) lAnimation.CurrentTime -= lAnimation.Duration;

                    for( auto &lChannel : lAnimation.mChannels )
                    {
                        auto &lAnimatedTransform = lChannel.mTargetNode.Get<AnimatedTransformComponent>();

                        float dt = lAnimation.CurrentTime - lChannel.mInterpolation.mInputs[lAnimation.CurrentTick];
                        if( lAnimation.CurrentTime >
                            lChannel.mInterpolation
                                .mInputs[( lAnimation.CurrentTick + 1 ) % lChannel.mInterpolation.mInputs.size()] )
                        {
                            lAnimation.CurrentTick += 1;
                            lAnimation.CurrentTick %= lChannel.mInterpolation.mInputs.size();
                            dt = lAnimation.CurrentTime - lChannel.mInterpolation.mInputs[lAnimation.CurrentTick];
                        }

                        switch( lChannel.mChannelID )
                        {
                        case sImportedAnimationChannel::Channel::ROTATION:
                        {
                            glm::quat q1;
                            q1.x = lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick].x;
                            q1.y = lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick].y;
                            q1.z = lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick].z;
                            q1.w = lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick].w;

                            glm::quat q2;
                            q2.x = lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick + 1].x;
                            q2.y = lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick + 1].y;
                            q2.z = lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick + 1].z;
                            q2.w = lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick + 1].w;

                            lAnimatedTransform.Rotation = glm::normalize( glm::slerp( q1, q2, dt ) );
                            break;
                        }
                        case sImportedAnimationChannel::Channel::TRANSLATION:
                        {
                            lAnimatedTransform.Translation =
                                math::vec3( glm::mix( lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick],
                                    lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick + 1], dt ) );
                            break;
                        }
                        case sImportedAnimationChannel::Channel::SCALE:
                        {
                            lAnimatedTransform.Scaling =
                                math::vec3( glm::mix( lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick],
                                    lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick + 1], dt ) );
                            break;
                        }
                        }
                    }
                } );

            ForEach<AnimatedTransformComponent>(
                [&]( auto l_ElementToProcess, auto &lAnimatedTransform )
                {
                    auto lMatrix = glm::translate( glm::mat4( 1.0f ), lAnimatedTransform.Translation ) *
                                   glm::mat4( lAnimatedTransform.Rotation ) *
                                   glm::scale( glm::mat4( 1.0f ), lAnimatedTransform.Scaling );
                    l_ElementToProcess.AddOrReplace<LocalTransformComponent>( lMatrix );
                } );
        }

        ForEach<SkeletonComponent, TransformMatrixComponent>(
            [&]( auto l_ElementToProcess, auto &s, auto &t )
            {
                math::mat4 lInverseTransform = math::Inverse( l_ElementToProcess.Get<TransformMatrixComponent>().Matrix );

                for( uint32_t lJointID = 0; lJointID < l_ElementToProcess.Get<SkeletonComponent>().Bones.size(); lJointID++ )
                {
                    Element    lJoint = l_ElementToProcess.Get<SkeletonComponent>().Bones[lJointID];
                    math::mat4 lInverseBindMatrix =
                        l_ElementToProcess.Get<SkeletonComponent>().InverseBindMatrices[lJointID];
                    math::mat4 lJointMatrix = lJoint.TryGet<TransformMatrixComponent>( TransformMatrixComponent{} ).Matrix;
                    lJointMatrix            = lInverseTransform * lJointMatrix * lInverseBindMatrix;

                    l_ElementToProcess.Get<SkeletonComponent>().JointMatrices[lJointID] = lJointMatrix;
                }
            } );

        if( mVertexBuffer )
        {
            LTSE_PROFILE_SCOPE( "Transform Vertices" );

            // Update the transformed vertex buffer for static meshies
            {
                std::vector<uint32_t>   lVertexOffsets{};
                std::vector<uint32_t>   lVertexCounts{};
                std::vector<math::mat4> lObjectToWorldTransforms{};
                uint32_t                lMaxVertexCount = 0;
                ForEach<StaticMeshComponent, TransformMatrixComponent>(
                    [&]( auto aEntiy, auto &aMesh, auto &aTransform )
                    {
                        if( aEntiy.Has<SkeletonComponent>() ) return;

                        lObjectToWorldTransforms.push_back( aTransform.Matrix );
                        lVertexOffsets.push_back( aMesh.mVertexOffset );
                        lVertexCounts.push_back( aMesh.mVertexCount );
                        lMaxVertexCount = std::max( lMaxVertexCount, static_cast<uint32_t>( aMesh.mVertexCount ) );
                    } );

                mTransforms.Upload( lObjectToWorldTransforms );
                mVertexOffsets.Upload( lVertexOffsets );
                mVertexCounts.Upload( lVertexCounts );

                StaticVertexTransform( mTransformedVertexBufferMemoryHandle.DataAs<VertexData>(),
                    mVertexBufferMemoryHandle.DataAs<VertexData>(), mTransforms.DataAs<math::mat4>(), lVertexOffsets.size(),
                    mVertexOffsets.DataAs<uint32_t>(), mVertexCounts.DataAs<uint32_t>(), lMaxVertexCount );
            }

            // Update the transformed vertex buffer for animated meshies
            {
                std::vector<uint32_t>   lVertexOffsets{};
                std::vector<uint32_t>   lVertexCounts{};
                std::vector<math::mat4> lObjectToWorldTransforms{};
                std::vector<math::mat4> lJointTransforms{};
                std::vector<uint32_t>   lJointOffsets{};
                uint32_t                lMaxVertexCount = 0;
                ForEach<StaticMeshComponent, TransformMatrixComponent, SkeletonComponent>(
                    [&]( auto aEntiy, auto &aMesh, auto &aTransform, auto &aSkeleton )
                    {
                        lObjectToWorldTransforms.push_back( aTransform.Matrix );
                        lVertexOffsets.push_back( aMesh.mVertexOffset );
                        lVertexCounts.push_back( aMesh.mVertexCount );
                        lMaxVertexCount = std::max( lMaxVertexCount, static_cast<uint32_t>( aMesh.mVertexCount ) );

                        lJointOffsets.push_back( lJointTransforms.size() );
                        for( auto &lJoint : aSkeleton.JointMatrices ) lJointTransforms.push_back( lJoint );
                    } );

                mTransforms.Upload( lObjectToWorldTransforms );
                mVertexOffsets.Upload( lVertexOffsets );
                mVertexCounts.Upload( lVertexCounts );
                mJointOffsets.Upload( lJointOffsets );
                mJointTransforms.Upload( lJointTransforms );

                SkinnedVertexTransform( mTransformedVertexBufferMemoryHandle.DataAs<VertexData>(),
                    mVertexBufferMemoryHandle.DataAs<VertexData>(), mTransforms.DataAs<math::mat4>(),
                    mJointTransforms.DataAs<math::mat4>(), mJointOffsets.DataAs<uint32_t>(), lVertexOffsets.size(),
                    mVertexOffsets.DataAs<uint32_t>(), mVertexCounts.DataAs<uint32_t>(), lMaxVertexCount );
            }

            CUDA_SYNC_CHECK();
        }

        UpdateRayTracingComponents();
    }

    void Scene::UpdateRayTracingComponents()
    {
        LTSE_PROFILE_FUNCTION();

        bool l_RebuildAS = false;
        ForEach<TransformMatrixComponent, StaticMeshComponent, RayTracingTargetComponent>(
            [&]( auto a_Entity, auto &a_TransformComponent, auto &a_MeshComponent, auto &a_RTComponent )
            {
                if( !glm::all( glm::equal( a_TransformComponent.Matrix, a_RTComponent.Transform, 0.0001 ) ) )
                {
                    a_RTComponent.Transform = a_TransformComponent.Matrix;
                    l_RebuildAS             = true;
                }
            } );

        if( l_RebuildAS ) RebuildAccelerationStructure();
    }

    void Scene::RebuildAccelerationStructure()
    {
        LTSE_PROFILE_FUNCTION();

        m_AccelerationStructure = LTSE::Core::New<OptixTraversableObject>( m_RayTracingContext );

        ForEach<RayTracingTargetComponent, StaticMeshComponent>(
            [&]( auto a_Entity, auto &a_RTComponent, auto &a_MeshComponent )
            {
                m_AccelerationStructure->AddGeometry( mTransformedVertexBufferMemoryHandle, mIndexBufferMemoryHandle,
                    a_MeshComponent.mVertexOffset, a_MeshComponent.mVertexCount, a_MeshComponent.mIndexOffset,
                    a_MeshComponent.mIndexCount );
            } );

        m_AccelerationStructure->Build();
    }

    void Scene::InitializeRayTracing()
    {
        m_RayTracingContext     = LTSE::Core::New<OptixDeviceContextObject>();
        m_AccelerationStructure = LTSE::Core::New<OptixTraversableObject>( m_RayTracingContext );
    }

    void Scene::Render() {}

    void SaveAs( fs::path aPath ) {}

} // namespace LTSE::Core
