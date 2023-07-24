#include <algorithm>
#include <execution>
#include <filesystem>
#include <fmt/core.h>
#include <future>
#include <gli/gli.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <stack>
#include <unordered_map>

#include "yaml-cpp/yaml.h"

#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/Resource.h"

#include "Core/CUDA/Array/CudaBuffer.h"

#include "Graphics/Vulkan/VkPipeline.h"

#include "Components.h"
#include "Scene.h"
#include "VertexTransform.h"

#include "Renderer/MeshRenderer.h"

#include "Scene/Components/VisualHelpers.h"
#include "Serialize/AssetFile.h"
#include "Serialize/FileIO.h"
#include "Serialize/SerializeComponents.h"

namespace SE::Core
{
    namespace fs = std::filesystem;
    using namespace SE::Graphics;
    using namespace SE::Cuda;
    using namespace SE::Core::EntityComponentSystem;
    using namespace SE::Core::EntityComponentSystem::Components;

    Scene::Scene( Ref<IGraphicContext> aGraphicContext, Ref<SE::Core::UIContext> aUI )
        : mGraphicContext{ aGraphicContext }
    {
        mMaterialSystem    = New<MaterialSystem>( mGraphicContext );
        mNewMaterialSystem = New<NewMaterialSystem>( mGraphicContext );

        mEditorView = Translate( mat4( 1.0f ), -vec3( 0.0f, 1.0f, 7.5f ) );

        DefaultCamera                = mRegistry.CreateEntity( "DefaultCamera" );
        auto &lCameraComponent       = DefaultCamera.Add<sCameraComponent>();
        lCameraComponent.Position    = vec3{ 0.0f, 0.0f, 0.0f };
        lCameraComponent.Pitch       = 0.0f;
        lCameraComponent.Yaw         = 0.0f;
        lCameraComponent.Roll        = 0.0f;
        lCameraComponent.Near        = 0.001;
        lCameraComponent.Far         = 1000.0f;
        lCameraComponent.FieldOfView = 90.0f;
        lCameraComponent.AspectRatio = 16.0f / 9.0f;

        CurrentCamera = DefaultCamera;

        Environment = mRegistry.CreateEntity( "Environment" );
        Environment.Add<sAmbientLightingComponent>();
        Environment.Add<sBackgroundComponent>();

        Root = mRegistry.CreateEntityWithRelationship( "WorldRoot" );
        Root.Add<sNodeTransformComponent>();

        InitializeRayTracing();
        ConnectSignalHandlers();
    }

    template <typename _Component>
    static void CopyComponent( Entity &aSource, Entity &aDestination )
    {
        if( ( aSource.Has<_Component>() ) )
            aDestination.AddOrReplace<_Component>( aSource.Get<_Component>() );
    }

    Scene::Scene( Ref<Scene> aSource )
    {
        mGraphicContext = aSource->mGraphicContext;
        mMaterialSystem = aSource->mMaterialSystem;
        mEditorView     = aSource->mEditorView;

        InitializeRayTracing();
        ConnectSignalHandlers();

        std::unordered_map<std::string, Entity> lSourceEntities{};
        std::unordered_map<std::string, Entity> lClonedEntities{};

        aSource->ForEach<sUUID>(
            [&]( auto aEntity, auto &aUUID )
            {
                auto lClonedEntity = CreateEntity();

                CopyComponent<sUUID>( aEntity, lClonedEntity );
                CopyComponent<sTag>( aEntity, lClonedEntity );

                lSourceEntities[aUUID.mValue.str()] = aEntity;
                lClonedEntities[aUUID.mValue.str()] = lClonedEntity;
            } );

        // Copy simple components
        for( auto &[lUUID, lEntity] : lSourceEntities )
        {
            auto lClonedEntity = lClonedEntities[lUUID];

            CopyComponent<sNodeTransformComponent>( lEntity, lClonedEntity );
            CopyComponent<sAnimatedTransformComponent>( lEntity, lClonedEntity );

            CopyComponent<sStaticMeshComponent>( lEntity, lClonedEntity );
            CopyComponent<sWireframeComponent>( lEntity, lClonedEntity );
            CopyComponent<sWireframeMeshComponent>( lEntity, lClonedEntity );
            CopyComponent<sBoundingBoxComponent>( lEntity, lClonedEntity );

            CopyComponent<sParticleSystemComponent>( lEntity, lClonedEntity );
            CopyComponent<sParticleShaderComponent>( lEntity, lClonedEntity );

            CopyComponent<sRayTracingTargetComponent>( lEntity, lClonedEntity );

            CopyComponent<sMaterialComponent>( lEntity, lClonedEntity );
            CopyComponent<sMaterialShaderComponent>( lEntity, lClonedEntity );

            CopyComponent<sBackgroundComponent>( lEntity, lClonedEntity );

            CopyComponent<sAmbientLightingComponent>( lEntity, lClonedEntity );
            CopyComponent<sLightComponent>( lEntity, lClonedEntity );

            CopyComponent<sBehaviourComponent>( lEntity, lClonedEntity );
            if( ( lEntity.Has<sActorComponent>() ) )
            {
                auto &lNewScriptComponent = lClonedEntity.AddOrReplace<sActorComponent>( lEntity.Get<sActorComponent>() );

                lNewScriptComponent.Initialize( lClonedEntity );
            }

            CopyComponent<sUIComponent>( lEntity, lClonedEntity );

            CopyComponent<PointLightHelperComponent>( lEntity, lClonedEntity );
            CopyComponent<DirectionalLightHelperComponent>( lEntity, lClonedEntity );
            CopyComponent<FieldOfViewHelperComponent>( lEntity, lClonedEntity );
            CopyComponent<CameraHelperComponent>( lEntity, lClonedEntity );

            CopyComponent<sSkeletonComponent>( lEntity, lClonedEntity );
            if( lClonedEntity.Has<sSkeletonComponent>() )
            {
                auto &lSkeletonComponent = lClonedEntity.Get<sSkeletonComponent>();
                for( uint32_t i = 0; i < lSkeletonComponent.BoneCount; i++ )
                    lSkeletonComponent.Bones[i] = lClonedEntities[lSkeletonComponent.Bones[i].Get<sUUID>().mValue.str()];
            }

            CopyComponent<sAnimationComponent>( lEntity, lClonedEntity );
            if( lClonedEntity.Has<sAnimationComponent>() )
            {
                auto &lAnimationComponent = lClonedEntity.Get<sAnimationComponent>();
                for( auto &lChannel : lAnimationComponent.mChannels )
                    lChannel.mTargetNode = lClonedEntities[lChannel.mTargetNode.Get<sUUID>().mValue.str()];
            }

            CopyComponent<sAnimationChooser>( lEntity, lClonedEntity );
            if( lClonedEntity.Has<sAnimationChooser>() )
            {
                auto &lAnimationChooser = lClonedEntity.Get<sAnimationChooser>();
                for( uint32_t i = 0; i < lAnimationChooser.Animations.size(); i++ )
                    lAnimationChooser.Animations[i] = lClonedEntities[lAnimationChooser.Animations[i].Get<sUUID>().mValue.str()];
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
                    auto &lDestParentEntity = lClonedEntities[lSourceParentEntity.Get<sUUID>().mValue.str()];

                    mRegistry.SetParent( lDestEntity, lDestParentEntity );
                }
            }
        }

        Root = lClonedEntities[aSource->Root.Get<sUUID>().mValue.str()];

        Environment = lClonedEntities[aSource->Environment.Get<sUUID>().mValue.str()];

        DefaultCamera = lClonedEntities[aSource->DefaultCamera.Get<sUUID>().mValue.str()];
        CurrentCamera = lClonedEntities[aSource->CurrentCamera.Get<sUUID>().mValue.str()];

        ResizeCUDABuffers();

        mIsClone = true;
    }

    Scene::~Scene()
    {
        mTransforms.Dispose();
        mVertexOffsets.Dispose();
        mVertexCounts.Dispose();
    }

    void Scene::SetViewport( vec2 aPosition, vec2 aSize )
    {
        mViewportPosition = aPosition;
        mViewportSize     = aSize;
    }

    void Scene::ResizeCUDABuffers()
    {
        const std::array<GPUMemory *, 7> lBuffers = { &mTransforms,   &mVertexBuffers,   &mTransformedBuffers, &mVertexOffsets,
                                                      &mVertexCounts, &mJointTransforms, &mJointOffsets };
        for( auto const &lP : lBuffers )
            lP->Dispose();

        uint32_t lTransformCount = 0;
        ForEach<sNodeTransformComponent>( [&]( auto aEntity, auto &aUUID ) { lTransformCount++; } );
        mTransforms = GPUMemory::Create<mat4>( lTransformCount );

        uint32_t lStaticMeshCount = 0;
        ForEach<sStaticMeshComponent>( [&]( auto aEntity, auto &aUUID ) { lStaticMeshCount++; } );
        mVertexBuffers      = GPUMemory::Create<VkGpuBuffer>( lStaticMeshCount );
        mTransformedBuffers = GPUMemory::Create<VkGpuBuffer>( lStaticMeshCount );
        mVertexOffsets      = GPUMemory::Create<uint32_t>( lStaticMeshCount );
        mVertexCounts       = GPUMemory::Create<uint32_t>( lStaticMeshCount );

        uint32_t lJointMatrixCount = 0;
        uint32_t lJointOffsetCount = 0;
        ForEach<sSkeletonComponent>(
            [&]( auto lElementToProcess, auto &s )
            {
                lJointMatrixCount += s.JointMatrices.size();
                lJointOffsetCount += 1;
            } );
        mJointTransforms = GPUMemory::Create<mat4>( lJointMatrixCount );
        mJointOffsets    = GPUMemory::Create<uint32_t>( lJointOffsetCount );
    }

    mat4 Scene::GetView()
    {
        mat4 lCameraView( 1.0f );
        if( CurrentCamera.Has<sCameraComponent>() )
        {
            auto &lComponent = CurrentCamera.Get<sCameraComponent>();

            mat4 lRx = Rotation( lComponent.Pitch, vec3( 1.0f, 0.0f, 0.0f ) );
            mat4 lRy = Rotation( lComponent.Yaw, vec3( 0.0f, 1.0f, 0.0f ) );
            mat4 lRz = Rotation( -lComponent.Roll, vec3( 0.0f, 0.0f, 1.0f ) );

            lCameraView = Inverse( Translate( lRx * lRy * lRz, lComponent.Position ) );
        }

        return lCameraView;
    }

    mat4 Scene::GetProjection()
    {
        mat4 l_CameraProjection( 1.0f );
        if( CurrentCamera.Has<sCameraComponent>() )
        {
            auto &lComponent = CurrentCamera.Get<sCameraComponent>();
            l_CameraProjection =
                Perspective( radians( lComponent.FieldOfView ), lComponent.AspectRatio, lComponent.Near, lComponent.Far );
            l_CameraProjection[1][1] *= -1.0f;
        }
        return l_CameraProjection;
    }

    vec3 Scene::GetCameraPosition()
    {
        vec3 l_CameraPosition( 0.0f );
        if( CurrentCamera.Has<sCameraComponent>() )
        {
            auto &lComponent = CurrentCamera.Get<sCameraComponent>();
            l_CameraPosition = lComponent.Position;
        }
        return l_CameraPosition;
    }

    mat4 Scene::GetFinalTransformMatrix( Element aElement )
    {
        auto &lUUID = aElement.Get<sUUID>().mValue;
        if( mTransformCache.find( lUUID ) != mTransformCache.end() )
            return mTransformCache[lUUID];
        return mat4( 1.0f );
    }

    Scene::Element Scene::Create( std::string a_Name, Element a_Parent )
    {
        return mRegistry.CreateEntity( a_Parent, a_Name );
    }

    Scene::Element Scene::CreateEntity()
    {
        return mRegistry.CreateEntity();
    }

    Scene::Element Scene::CreateEntity( std::string a_Name )
    {
        return mRegistry.CreateEntity( a_Name );
    }

    void Scene::ClearScene()
    {
        mMaterialSystem->Wipe();
        mRegistry.Clear();

        mMaterialSystem->Clear();

        DefaultCamera                = mRegistry.CreateEntity( "DefaultCamera" );
        auto &lCameraComponent       = DefaultCamera.Add<sCameraComponent>();
        lCameraComponent.Position    = vec3{ 0.0f, 0.0f, 0.0f };
        lCameraComponent.Pitch       = 0.0f;
        lCameraComponent.Yaw         = 0.0f;
        lCameraComponent.Roll        = 0.0f;
        lCameraComponent.Near        = 0.001;
        lCameraComponent.Far         = 1000.0f;
        lCameraComponent.FieldOfView = 90.0f;
        lCameraComponent.AspectRatio = 16.0f / 9.0f;

        CurrentCamera = DefaultCamera;

        Environment = mRegistry.CreateEntity( "Environment" );
        Environment.Add<sAmbientLightingComponent>();
        Environment.Add<sBackgroundComponent>();

        Root = mRegistry.CreateEntityWithRelationship( "WorldRoot" );
        Root.Add<sNodeTransformComponent>();
    }

    void Scene::ConnectSignalHandlers()
    {
        // clang-format off
        mRegistry.OnComponentAdded<sActorComponent>( 
            [&]( auto aEntity, auto &aComponent ) { 
                aComponent.Initialize( aEntity ); 
        } );

        mRegistry.OnComponentUpdated<sActorComponent>( 
            [&]( auto aEntity, auto &aComponent ) { 
                aComponent.Initialize( aEntity ); 
        } );

        mRegistry.OnComponentAdded<sUIComponent>( [&]( auto aEntity, auto &aComponent ) { 
                aComponent.Initialize( aEntity ); 
        } );

        mRegistry.OnComponentUpdated<sUIComponent>( [&]( auto aEntity, auto &aComponent ) { 
                aComponent.Initialize( aEntity ); 
        } );
        // clang-format on
    }

    Scene::Element Scene::LoadModel( Ref<sImportedModel> aModelData, mat4 aTransform )
    {
        return LoadModel( aModelData, aTransform, "MODEL" );
    }

    void Scene::LoadScenario( fs::path aScenarioPath )
    {
        mRegistry.Clear();
        mMaterialSystem->Wipe();

        ConnectSignalHandlers();

        auto lScenarioRoot = aScenarioPath.parent_path();

        std::vector<sImportedAnimationSampler> lInterpolationData;
        {
            BinaryAsset lBinaryDataFile( lScenarioRoot / "Animations" / "BinaryData.bin" );

            for( uint32_t lInterpolationIndex = 0; lInterpolationIndex < lBinaryDataFile.CountAssets(); lInterpolationIndex++ )
            {
                auto &lAnimationData = lInterpolationData.emplace_back();

                lBinaryDataFile.Retrieve( lInterpolationIndex, lAnimationData );
            }
        }

        auto lRootNode = YAML::LoadFile( aScenarioPath.string() );

        sReadContext                                 lReadContext{};
        std::unordered_map<std::string, std::string> lParentEntityLUT{};

        auto &lSceneRoot = lRootNode["scene"];
        auto &lNodesRoot = lSceneRoot["nodes"];

        std::vector<std::tuple<std::string, sStaticMeshComponent, std::string>> lBufferLoadQueue{};

        std::map<std::string, std::set<std::string>> lMaterialLoadQueue{};
        for( YAML::iterator it = lNodesRoot.begin(); it != lNodesRoot.end(); ++it )
        {
            auto const &lKey                 = it->first.as<std::string>();
            auto       &lEntityConfiguration = it->second;

            auto lEntity = mRegistry.CreateEntity( sUUID( lKey ) );
            auto lUUID   = lEntity.Get<sUUID>().mValue;

            lReadContext.mEntities[lKey] = lEntity;

            if( HasTypeTag<sRelationshipComponent>( lEntityConfiguration ) )
            {
                if( !( lEntityConfiguration[TypeTag<sRelationshipComponent>()]["mParent"].IsNull() ) )
                {
                    auto lParentUUIDStr = Get( lEntityConfiguration[TypeTag<sRelationshipComponent>()]["mParent"], std::string{ "" } );

                    auto lParentUUID       = UUIDv4::UUID::fromStrFactory( lParentUUIDStr );
                    lParentEntityLUT[lKey] = lParentUUIDStr;
                }
            }

            if( HasTypeTag<sStaticMeshComponent>( lEntityConfiguration ) )
            {
                auto &[lEntityID, lComponent, lBufferID] = lBufferLoadQueue.emplace_back();

                lEntityID = lKey;
                lBufferID = Get( lEntityConfiguration[TypeTag<sStaticMeshComponent>()]["mMeshData"], std::string{ "" } );
                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sStaticMeshComponent>()], lReadContext );
            }

            if( HasTypeTag<sMaterialComponent>( lEntityConfiguration ) )
            {
                auto &lMaterialID = Get( lEntityConfiguration[TypeTag<sMaterialComponent>()]["mMaterialPath"], std::string{ "" } );

                if( lMaterialLoadQueue.find( lMaterialID ) == lMaterialLoadQueue.end() )
                    lMaterialLoadQueue.emplace( lMaterialID, std::set<std::string>{ lKey } );
                else
                    lMaterialLoadQueue[lMaterialID].emplace( lKey );
            }

            if( HasTypeTag<sTag>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sTag>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sTag>()], lReadContext );
            }

            if( HasTypeTag<sCameraComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sCameraComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sCameraComponent>()], lReadContext );
            }

            if( HasTypeTag<sAnimatedTransformComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sAnimatedTransformComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sAnimatedTransformComponent>()], lReadContext );
            }

            if( HasTypeTag<sNodeTransformComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sNodeTransformComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sNodeTransformComponent>()], lReadContext );
            }

            if( HasTypeTag<sRayTracingTargetComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sRayTracingTargetComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sRayTracingTargetComponent>()], lReadContext );
            }

            if( HasTypeTag<sMaterialShaderComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sMaterialShaderComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sMaterialShaderComponent>()], lReadContext );
            }

            if( HasTypeTag<sBackgroundComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sBackgroundComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sBackgroundComponent>()], lReadContext );
            }

            if( HasTypeTag<sAmbientLightingComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sAmbientLightingComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sAmbientLightingComponent>()], lReadContext );
            }

            if( HasTypeTag<sLightComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sLightComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sLightComponent>()], lReadContext );
            }

            if( HasTypeTag<sActorComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sActorComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sActorComponent>()], lReadContext );
            }

            if( HasTypeTag<sUIComponent>( lEntityConfiguration ) )
            {
                sUIComponent lComponent{};

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sUIComponent>()], lReadContext );
                lEntity.Add<sUIComponent>( lComponent );
            }
        }

        std::mutex lMaterialSystemLock;
        uint32_t   lMaterialIndex = 0;
        // clang-format off
        std::for_each( std::execution::seq, lMaterialLoadQueue.begin(), lMaterialLoadQueue.end(), [&]( auto const &aElement )
        {
            auto &[lMaterialID, lEntities] = aElement;

            auto &lNewMaterial = mMaterialSystem->CreateMaterial( lScenarioRoot / lMaterialID );
            auto &lNewMaterial1 = mNewMaterialSystem->CreateMaterial( lScenarioRoot / lMaterialID );

            for( auto &lEntity : lEntities )
            {
                lReadContext.mEntities[lEntity].AddOrReplace<sMaterialComponent>( lNewMaterial.mID );
                lReadContext.mEntities[lEntity].AddOrReplace<sNewMaterialComponent>( lNewMaterial1 );
            }
        } );

        mNewMaterialSystem->UpdateMaterialData();

        std::mutex lBufferLock;
        // clang-format on

        // clang-format off
        std::for_each( std::execution::seq, lBufferLoadQueue.begin(), lBufferLoadQueue.end(), [&]( auto &aElement )
        {
            auto &[lEntityID, lComponent, lBufferID] = aElement;

            BinaryAsset lBinaryDataFile( lScenarioRoot / lBufferID );

            std::vector<VertexData> lVertexBuffer;
            std::vector<uint32_t>   lIndexBuffer;
            lBinaryDataFile.Retrieve( 0, lVertexBuffer, lIndexBuffer );

            lComponent.mVertexBuffer =
                CreateBuffer( mGraphicContext, lVertexBuffer, eBufferType::VERTEX_BUFFER, false, false, true, true );
            lComponent.mIndexBuffer =
                CreateBuffer( mGraphicContext, lIndexBuffer, eBufferType::INDEX_BUFFER, false, false, true, true );
            lComponent.mTransformedBuffer = CreateBuffer( mGraphicContext, eBufferType::VERTEX_BUFFER, false, false,
                                                            true, true, lComponent.mVertexBuffer->SizeAs<uint8_t>() );

            lReadContext.mEntities[lEntityID].AddOrReplace<sStaticMeshComponent>( lComponent );
        } );
        // clang-format on

        for( YAML::iterator it = lNodesRoot.begin(); it != lNodesRoot.end(); ++it )
        {
            auto const &lKey                 = it->first.as<std::string>();
            auto       &lEntityConfiguration = it->second;

            auto &lEntity = lReadContext.mEntities[lKey];

            if( !lEntity )
                return;

            if( lParentEntityLUT.find( lKey ) != lParentEntityLUT.end() )
                mRegistry.SetParent( lEntity, lReadContext.mEntities[lParentEntityLUT[lKey]] );
        }

        for( YAML::iterator it = lNodesRoot.begin(); it != lNodesRoot.end(); ++it )
        {
            auto const &lKey                 = it->first.as<std::string>();
            auto       &lEntityConfiguration = it->second;

            auto &lEntity = lReadContext.mEntities[lKey];

            if( !lEntity )
                return;

            if( HasTypeTag<sSkeletonComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sSkeletonComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sSkeletonComponent>()], lReadContext );
            }
        }

        for( YAML::iterator it = lNodesRoot.begin(); it != lNodesRoot.end(); ++it )
        {
            auto const &lKey                 = it->first.as<std::string>();
            auto       &lEntityConfiguration = it->second;

            auto &lEntity = lReadContext.mEntities[lKey];

            if( !lEntity )
                return;

            if( HasTypeTag<sAnimationComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sAnimationComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sAnimationComponent>()], lReadContext, lInterpolationData );
            }
        }

        for( YAML::iterator it = lNodesRoot.begin(); it != lNodesRoot.end(); ++it )
        {
            auto const &lKey                 = it->first.as<std::string>();
            auto       &lEntityConfiguration = it->second;

            auto &lEntity = lReadContext.mEntities[lKey];

            if( !lEntity )
                return;

            if( HasTypeTag<sAnimationChooser>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sAnimationChooser>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sAnimationChooser>()], lReadContext );
            }
        }

        mEditorView = ReadMatrix( lSceneRoot["editor_view"] );

        auto lRootNodeUUIDStr = Get( lSceneRoot["root"], std::string{ "" } );
        auto lRootNodeUUID    = UUIDv4::UUID::fromStrFactory( lRootNodeUUIDStr );
        Root                  = lReadContext.mEntities[lRootNodeUUIDStr];
        SE::Logging::Info( "Created root", lRootNodeUUIDStr );

        auto lEnvironmentNodeUUID = Get( lSceneRoot["environment"], std::string{ "" } );
        Environment               = lReadContext.mEntities[lEnvironmentNodeUUID];
        SE::Logging::Info( "Created environment", lEnvironmentNodeUUID );

        auto lCurrentCameraUUID = Get( lSceneRoot["current_camera"], std::string{ "" } );
        CurrentCamera           = lReadContext.mEntities[lCurrentCameraUUID];
        SE::Logging::Info( "Created camera", lCurrentCameraUUID );

        auto lDefaultCameraUUID = Get( lSceneRoot["default_camera"], std::string{ "" } );
        DefaultCamera           = lReadContext.mEntities[lDefaultCameraUUID];
        SE::Logging::Info( "Created camera", lDefaultCameraUUID );

        ResizeCUDABuffers();

        RebuildAccelerationStructure();
    }

    Scene::Element Scene::LoadModel( Ref<sImportedModel> aModelData, mat4 aTransform, std::string a_Name )
    {
        auto lAssetEntity = mRegistry.CreateEntity( Root, a_Name );
        lAssetEntity.Add<sNodeTransformComponent>( aTransform );

        std::vector<uint32_t> lTextureIds = {};
        for( auto &lTexture : aModelData->mTextures )
            lTextureIds.push_back( mMaterialSystem->CreateTexture( lTexture.mTexture, lTexture.mSampler ) );

        std::vector<uint32_t>                 lMaterialIds        = {};
        std::vector<sMaterialShaderComponent> lMaterialCreateInfo = {};
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

            lNewMaterial.mRoughnessFactor              = lMaterial.mConstants.mRoughnessFactor;
            lNewMaterial.mMetallicFactor               = lMaterial.mConstants.mMetallicFactor;
            lNewMaterial.mMetalRoughTexture.mTextureID = lGetTexID( lMaterial.mTextures.mMetallicRoughnessTexture.TextureID, 0 );
            lNewMaterial.mMetalRoughTexture.mUVChannel = lMaterial.mTextures.mMetallicRoughnessTexture.UVChannel;

            lNewMaterial.mOcclusionStrength           = 0.0f;
            lNewMaterial.mOcclusionTexture.mTextureID = lGetTexID( lMaterial.mTextures.mOcclusionTexture.TextureID, 1 );
            lNewMaterial.mOcclusionTexture.mUVChannel = lMaterial.mTextures.mOcclusionTexture.UVChannel;

            lNewMaterial.mNormalsTexture.mTextureID = lGetTexID( lMaterial.mTextures.mNormalTexture.TextureID, 0 );
            lNewMaterial.mNormalsTexture.mUVChannel = lMaterial.mTextures.mNormalTexture.UVChannel;
            lMaterialIds.push_back( lNewMaterial.mID );

            sMaterialShaderComponent lMaterialShader{};
            lMaterialShader.Type              = eMaterialType::Opaque;
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
            sStaticMeshComponent lMeshComponent{};
            lMeshComponent.mName      = lMesh.mName;
            lMeshComponent.mPrimitive = lMesh.mPrimitive;

            std::vector<VertexData> lVertices( lMesh.mPositions.size() );
            for( uint32_t i = 0; i < lMesh.mPositions.size(); i++ )
            {
                lVertices[i].Position    = lMesh.mPositions[i];
                lVertices[i].Normal      = lMesh.mNormals[i];
                lVertices[i].TexCoords_0 = lMesh.mUV0[i];
                lVertices[i].Bones       = lMesh.mJoints[i];
                lVertices[i].Weights     = lMesh.mWeights[i];
            }

            lMeshComponent.mVertexBuffer =
                CreateBuffer( mGraphicContext, lVertices, eBufferType::VERTEX_BUFFER, false, false, true, true );
            lMeshComponent.mIndexBuffer =
                CreateBuffer( mGraphicContext, lMesh.mIndices, eBufferType::INDEX_BUFFER, false, false, true, true );
            lMeshComponent.mTransformedBuffer = CreateBuffer( mGraphicContext, eBufferType::VERTEX_BUFFER, false, false, true, true,
                                                              lMeshComponent.mVertexBuffer->SizeAs<uint8_t>() );

            lMeshComponent.mVertexOffset = 0;
            lMeshComponent.mVertexCount  = lVertices.size();
            lMeshComponent.mIndexOffset  = 0;
            lMeshComponent.mIndexCount   = lMesh.mIndices.size();

            auto lMeshEntity = Create( lMesh.mName, lAssetEntity );
            lMeshEntity.Add<sStaticMeshComponent>( lMeshComponent );
            lMeshEntity.Add<sMaterialComponent>( lMaterialIds[lMesh.mMaterialID] );
            lMeshEntity.Add<sMaterialShaderComponent>( lMaterialCreateInfo[lMesh.mMaterialID] );
            lMeshEntity.Add<sNodeTransformComponent>( mat4( 1.0f ) );

            lMeshes.push_back( lMeshEntity );
        }

        std::vector<Element> lNodes = {};
        for( auto &lNode : aModelData->mNodes )
        {
            auto lNodeEntity = mRegistry.CreateEntityWithRelationship( lNode.mName );
            lNodeEntity.Add<sNodeTransformComponent>( lNode.mTransform );

            lNodes.push_back( lNodeEntity );
        }

        int32_t lIndex = 0;
        for( auto &lNodeEntity : lNodes )
        {
            if( aModelData->mNodes[lIndex].mParentID == std::numeric_limits<uint32_t>::max() )
                mRegistry.SetParent( lNodeEntity, lAssetEntity );
            else
                mRegistry.SetParent( lNodeEntity, lNodes[aModelData->mNodes[lIndex].mParentID] );
            lIndex++;
        }

        for( uint32_t lNodeID = 0; lNodeID < aModelData->mNodes.size(); lNodeID++ )
        {
            auto &lNode = aModelData->mNodes[lNodeID];

            if( lNode.mMeshes.size() == 0 )
                continue;

            for( uint32_t lNodeMeshID = 0; lNodeMeshID < lNode.mMeshes.size(); lNodeMeshID++ )
                mRegistry.SetParent( lMeshes[lNodeMeshID], lNodes[lNodeID] );

            if( lNode.mSkinID != std::numeric_limits<uint32_t>::max() )
            {
                auto &lSkin = aModelData->mSkins[lNode.mSkinID];

                sSkeletonComponent lNodeSkeleton{};

                std::vector<Element> lJointNodes = {};
                for( uint32_t i = 0; i < lSkin.mJointNodeID.size(); i++ )
                {
                    lNodeSkeleton.Bones.push_back( lNodes[lSkin.mJointNodeID[i]] );
                    lNodeSkeleton.InverseBindMatrices.push_back( lSkin.mInverseBindMatrices[i] );
                    lNodeSkeleton.JointMatrices.push_back( mat4( 0.0f ) );
                }

                lNodeSkeleton.BoneCount = lNodeSkeleton.Bones.size();
                for( uint32_t i = 0; i < lNode.mMeshes.size(); i++ )
                {
                    auto &lMesh = lMeshes[lNode.mMeshes[i]];
                    lMesh.Add<sSkeletonComponent>( lNodeSkeleton );
                }
            }
        }

        if( aModelData->mAnimations.size() > 0 )
            lAssetEntity.Add<sAnimationChooser>();

        for( auto &lAnimation : aModelData->mAnimations )
        {
            auto &lAnimationChooser = lAssetEntity.Get<sAnimationChooser>();

            auto  l_AnimationEntity   = mRegistry.CreateEntity( lAssetEntity, lAnimation.mName );
            auto &lAnimationComponent = l_AnimationEntity.Add<sAnimationComponent>();

            lAnimationChooser.Animations.push_back( l_AnimationEntity );
            lAnimationComponent.Duration = lAnimation.mEnd - lAnimation.mStart;

            for( uint32_t lAnimationChannelIndex = 0; lAnimationChannelIndex < lAnimation.mChannels.size(); lAnimationChannelIndex++ )
            {
                sAnimationChannel lAnimationChannel{};
                lAnimationChannel.mChannelID     = lAnimation.mChannels[lAnimationChannelIndex].mComponent;
                lAnimationChannel.mInterpolation = lAnimation.mSamplers[lAnimation.mChannels[lAnimationChannelIndex].mSamplerIndex];
                lAnimationChannel.mTargetNode    = lNodes[lAnimation.mChannels[lAnimationChannelIndex].mNodeID];
                lAnimationChannel.mTargetNode.TryAdd<sAnimatedTransformComponent>();
                lAnimationChannel.mTargetNode.TryAdd<sStaticTransformComponent>(
                    lAnimationChannel.mTargetNode.Get<sNodeTransformComponent>().mMatrix );

                lAnimationComponent.mChannels.push_back( lAnimationChannel );
            }
        }

        ResizeCUDABuffers();

        return lAssetEntity;
    }

    void Scene::MarkAsRayTracingTarget( Scene::Element aElement )
    {
        if( !aElement.Has<sStaticMeshComponent>() )
            return;

        if( aElement.Has<sRayTracingTargetComponent>() )
            return;

        auto &lRTComponent = aElement.Add<sRayTracingTargetComponent>();
    }

    void Scene::AttachScript( Element aElement, std::string aClassName )
    {
        auto &lNewScriptComponent = aElement.Add<sActorComponent>( aClassName );
    }

    void Scene::BeginScenario()
    {
        if( mState != eSceneState::EDITING )
            return;

        ForEach<sAnimatedTransformComponent>(
            [=]( auto lEntity, auto &lComponent )
            { lEntity.AddOrReplace<sStaticTransformComponent>( lEntity.Get<sNodeTransformComponent>().mMatrix ); } );

        // Initialize native scripts
        ForEach<sBehaviourComponent>(
            [=]( auto lEntity, auto &lComponent )
            {
                if( !lComponent.ControllerInstance && lComponent.InstantiateController )
                {
                    lComponent.ControllerInstance = lComponent.InstantiateController();
                    lComponent.ControllerInstance->Initialize( mRegistry.WrapEntity( lEntity ) );
                    lComponent.ControllerInstance->OnBeginScenario();
                }
            } );

        ForEach<sActorComponent>( [=]( auto lEntity, auto &lComponent ) { lComponent.OnBeginScenario(); } );
        ForEach<sUIComponent>( [=]( auto lEntity, auto &lComponent ) { lComponent.OnBeginScenario(); } );

        mState = eSceneState::RUNNING;
    }

    void Scene::EndScenario()
    {
        if( mState != eSceneState::RUNNING )
            return;

        ForEach<sAnimatedTransformComponent>(
            [=]( auto lEntity, auto &lComponent )
            { lEntity.AddOrReplace<sNodeTransformComponent>( lEntity.Get<sStaticTransformComponent>().Matrix ); } );

        // Destroy scripts
        ForEach<sBehaviourComponent>(
            [=]( auto lEntity, auto &lComponent )
            {
                if( lComponent.ControllerInstance )
                {
                    lComponent.ControllerInstance->OnEndScenario();
                    lComponent.DestroyController( &lComponent );
                }
            } );

        // Destroy Lua scripts
        ForEach<sActorComponent>( [=]( auto lEntity, auto &lComponent ) { lComponent.OnEndScenario(); } );

        mState = eSceneState::EDITING;
    }

    void Scene::UpdateAnimation( Entity &aEntity, Timestep const &ts )
    {
        auto &lAnimation = aEntity.Get<sAnimationComponent>();

        lAnimation.CurrentTime += ( ts / 1000.0f );
        if( lAnimation.CurrentTime > lAnimation.Duration )
        {
            lAnimation.CurrentTime -= lAnimation.Duration;
            lAnimation.CurrentTick = 0;
        }

        for( auto &lChannel : lAnimation.mChannels )
        {
            auto &lAnimatedTransform = lChannel.mTargetNode.Get<sAnimatedTransformComponent>();
            if( lAnimation.CurrentTick >= lChannel.mInterpolation.mInputs.size() )
                continue;

            float dt = lAnimation.CurrentTime - lChannel.mInterpolation.mInputs[lAnimation.CurrentTick];
            if( lAnimation.CurrentTime >
                lChannel.mInterpolation.mInputs[( lAnimation.CurrentTick + 1 ) % lChannel.mInterpolation.mInputs.size()] )
            {
                lAnimation.CurrentTick += 1;
                lAnimation.CurrentTick %= ( lChannel.mInterpolation.mInputs.size() - 1 );
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
                    vec3( glm::mix( lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick],
                                    lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick + 1], dt ) );
                break;
            }
            case sImportedAnimationChannel::Channel::SCALE:
            {
                lAnimatedTransform.Scaling = vec3( glm::mix( lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick],
                                                             lChannel.mInterpolation.mOutputsVec4[lAnimation.CurrentTick + 1], dt ) );
                break;
            }
            }
        }
    }

    void Scene::Update( Timestep ts )
    {
        SE_PROFILE_FUNCTION();

        // Run scripts if the scene is in RUNNING mode.  The native scripts are run first, followed by the Lua scripts.
        if( mState == eSceneState::RUNNING )
        {
            ForEach<sBehaviourComponent>(
                [=]( auto aEntity, auto &aComponent )
                {
                    if( aComponent.ControllerInstance )
                        aComponent.ControllerInstance->OnTick( ts );
                } );

            ForEach<sActorComponent>( [=]( auto lEntity, auto &lComponent ) { lComponent.OnTick( ts ); } );

            // Update animations
            ForEach<sAnimationChooser>( [=]( auto aEntity, auto &aComponent ) { UpdateAnimation( aComponent.Animations[0], ts ); } );

            ForEach<sAnimatedTransformComponent>(
                [&]( auto aEntity, auto &aComponent )
                {
                    mat4 lRotation    = mat4( aComponent.Rotation );
                    mat4 lTranslation = Translation( aComponent.Translation );
                    mat4 lScale       = Scaling( aComponent.Scaling );

                    aEntity.AddOrReplace<sNodeTransformComponent>( lTranslation * lRotation * lScale );
                } );
        }

        ForEach<sUIComponent>(
            [=]( auto aEntity, auto &aComponent )
            {
                if( ( mState != eSceneState::RUNNING ) && !aComponent.mDisplayInEditor )
                    return;

                static ImGuiWindowFlags lFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;

                ImVec2 lPosition = ImVec2{ aComponent.mX, aComponent.mY } + ImVec2{ mViewportPosition.x, mViewportPosition.y };

                ImGui::SetNextWindowPos( lPosition, ImGuiCond_Always );
                ImGui::SetNextWindowSize( ImVec2{ aComponent.mWidth, aComponent.mHeight }, ImGuiCond_Always );

                ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, aComponent.mRounding );
                ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, aComponent.mBorderThickness );

                auto &lFillColor = aComponent.mFillColor;
                ImGui::PushStyleColor( ImGuiCol_WindowBg, ImVec4{ lFillColor.x, lFillColor.y, lFillColor.z, lFillColor.w } );

                auto &lBorderColor = aComponent.mBorderColor;
                ImGui::PushStyleColor( ImGuiCol_Border, ImVec4{ lBorderColor.x, lBorderColor.y, lBorderColor.z, lBorderColor.w } );

                auto lWindowID = fmt::format( "##{}", aEntity.Get<sUUID>().mValue.str() );
                if( ImGui::Begin( lWindowID.c_str(), NULL, lFlags ) )
                {
                    // If the scene is in a running state, call the scripted function to populate the window
                    // and interact with it.
                    if( mState == eSceneState::RUNNING )
                        aComponent.OnUpdate( ts );
                    else if( aComponent.mPreview )
                        aComponent.OnPreviewUpdate( ts );
                }
                ImGui::End();
                ImGui::PopStyleColor( 2 );
                ImGui::PopStyleVar( 2 );
            } );

        mTransformCache.clear();
        ForEach<sUUID, sNodeTransformComponent>(
            [&]( auto aEntity, auto &aUUID, auto &aTransformComponent )
            {
                if( mTransformCache.find( aUUID.mValue ) != mTransformCache.end() )
                    return;

                mat4 lAccumulator = aTransformComponent.mMatrix;

                auto lCurrentEntity = aEntity.Get<sRelationshipComponent>().mParent;
                while( lCurrentEntity )
                {
                    lAccumulator   = lCurrentEntity.Get<sNodeTransformComponent>().mMatrix * lAccumulator;
                    lCurrentEntity = lCurrentEntity.Get<sRelationshipComponent>().mParent;
                }

                mTransformCache[aUUID.mValue] = lAccumulator;
            } );

        ForEach<sUUID, sSkeletonComponent>(
            [&]( auto lElementToProcess, auto &aUUID, auto &aSkeleton )
            {
                if( mTransformCache.find( aUUID.mValue ) == mTransformCache.end() )
                    return;
                mat4 lInverseTransform = Inverse( mTransformCache[aUUID.mValue] );

                for( uint32_t lJointID = 0; lJointID < lElementToProcess.Get<sSkeletonComponent>().Bones.size(); lJointID++ )
                {
                    Element lJoint             = lElementToProcess.Get<sSkeletonComponent>().Bones[lJointID];
                    mat4    lInverseBindMatrix = lElementToProcess.Get<sSkeletonComponent>().InverseBindMatrices[lJointID];
                    mat4    lJointMatrix       = mTransformCache[lJoint.Get<sUUID>().mValue];
                    lJointMatrix               = lInverseTransform * lJointMatrix * lInverseBindMatrix;

                    lElementToProcess.Get<sSkeletonComponent>().JointMatrices[lJointID] = lJointMatrix;
                }
            } );

        // Update the transformed vertex buffer for static meshies
        {
            std::vector<SE::Cuda::Internal::sGPUDevicePointerView> lVertexBuffers{};
            std::vector<SE::Cuda::Internal::sGPUDevicePointerView> lOutVertexBuffers{};

            std::vector<uint32_t> lVertexOffsets{};
            std::vector<uint32_t> lVertexCounts{};
            std::vector<mat4>     lObjectToWorldTransforms{};
            uint32_t              lMaxVertexCount = 0;
            ForEach<sUUID, sStaticMeshComponent>(
                [&]( auto aEntiy, auto &aUUID, auto &aMesh )
                {
                    if( aEntiy.Has<sSkeletonComponent>() )
                        return;
                    if( mTransformCache.find( aUUID.mValue ) == mTransformCache.end() )
                        return;

                    lObjectToWorldTransforms.push_back( mTransformCache[aUUID.mValue] );
                    lVertexBuffers.push_back( *aMesh.mVertexBuffer );
                    lOutVertexBuffers.push_back( *aMesh.mTransformedBuffer );
                    lVertexOffsets.push_back( aMesh.mVertexOffset );
                    lVertexCounts.push_back( aMesh.mVertexCount );
                    lMaxVertexCount = std::max( lMaxVertexCount, static_cast<uint32_t>( aMesh.mVertexCount ) );
                } );

            mTransforms.Upload( lObjectToWorldTransforms );
            mVertexOffsets.Upload( lVertexOffsets );
            mVertexCounts.Upload( lVertexCounts );
            mVertexBuffers.Upload( lVertexBuffers );
            mTransformedBuffers.Upload( lOutVertexBuffers );

            StaticVertexTransform( mTransformedBuffers.DataAs<SE::Cuda::Internal::sGPUDevicePointerView>(),
                                   mVertexBuffers.DataAs<SE::Cuda::Internal::sGPUDevicePointerView>(), mTransforms.DataAs<mat4>(),
                                   lVertexOffsets.size(), mVertexOffsets.DataAs<uint32_t>(), mVertexCounts.DataAs<uint32_t>(),
                                   lMaxVertexCount );
        }

        // Update the transformed vertex buffer for animated meshies
        {
            std::vector<SE::Cuda::Internal::sGPUDevicePointerView> lVertexBuffers{};
            std::vector<SE::Cuda::Internal::sGPUDevicePointerView> lOutVertexBuffers{};

            std::vector<uint32_t> lVertexOffsets{};
            std::vector<uint32_t> lVertexCounts{};
            std::vector<mat4>     lObjectToWorldTransforms{};
            std::vector<mat4>     lJointTransforms{};
            std::vector<uint32_t> lJointOffsets{};
            uint32_t              lMaxVertexCount = 0;
            ForEach<sUUID, sStaticMeshComponent, sSkeletonComponent>(
                [&]( auto aEntiy, auto &aUUID, auto &aMesh, auto &aSkeleton )
                {
                    if( mTransformCache.find( aUUID.mValue ) == mTransformCache.end() )
                        return;

                    lObjectToWorldTransforms.push_back( mTransformCache[aUUID.mValue] );
                    lVertexBuffers.push_back( *aMesh.mVertexBuffer );
                    lOutVertexBuffers.push_back( *aMesh.mTransformedBuffer );
                    lVertexOffsets.push_back( aMesh.mVertexOffset );
                    lVertexCounts.push_back( aMesh.mVertexCount );
                    lMaxVertexCount = std::max( lMaxVertexCount, static_cast<uint32_t>( aMesh.mVertexCount ) );

                    lJointOffsets.push_back( lJointTransforms.size() );
                    for( auto &lJoint : aSkeleton.JointMatrices )
                        lJointTransforms.push_back( lJoint );
                } );

            mTransforms.Upload( lObjectToWorldTransforms );
            mVertexBuffers.Upload( lVertexBuffers );
            mTransformedBuffers.Upload( lOutVertexBuffers );
            mVertexOffsets.Upload( lVertexOffsets );
            mVertexCounts.Upload( lVertexCounts );
            mJointOffsets.Upload( lJointOffsets );
            mJointTransforms.Upload( lJointTransforms );

            SkinnedVertexTransform( mTransformedBuffers.DataAs<SE::Cuda::Internal::sGPUDevicePointerView>(),
                                    mVertexBuffers.DataAs<SE::Cuda::Internal::sGPUDevicePointerView>(), mTransforms.DataAs<mat4>(),
                                    mJointTransforms.DataAs<mat4>(), mJointOffsets.DataAs<uint32_t>(), lVertexOffsets.size(),
                                    mVertexOffsets.DataAs<uint32_t>(), mVertexCounts.DataAs<uint32_t>(), lMaxVertexCount );
        }

        SyncDevice();

        UpdateRayTracingComponents();
    }

    void Scene::UpdateRayTracingComponents()
    {
        SE_PROFILE_FUNCTION();

        bool lRebuildAS = false;
        ForEach<sUUID, sStaticMeshComponent, sRayTracingTargetComponent>(
            [&]( auto aEntity, auto &aUUID, auto &aMeshComponent, auto &aRTComponent )
            {
                if( mTransformCache.find( aUUID.mValue ) == mTransformCache.end() )
                    return;

                if( !glm::all( glm::equal( mTransformCache[aUUID.mValue], aRTComponent.Transform, 0.0001 ) ) )
                {
                    aRTComponent.Transform = mTransformCache[aUUID.mValue];
                    lRebuildAS             = true;
                }
            } );

        if( lRebuildAS )
            RebuildAccelerationStructure();
    }

    void Scene::RebuildAccelerationStructure()
    {
        SE_PROFILE_FUNCTION();

        mAccelerationStructure = SE::Core::New<OptixScene>( mRayTracingContext );

        // ForEach<sRayTracingTargetComponent, sStaticMeshComponent>(
        //     [&]( auto aEntity, auto &aRTComponent, auto &aMeshComponent )
        //     {
        //         mAccelerationStructure->AddGeometry( *aMeshComponent.mTransformedBuffer, *aMeshComponent.mIndexBuffer,
        //                                              aMeshComponent.mVertexOffset, aMeshComponent.mVertexCount,
        //                                              aMeshComponent.mIndexOffset, aMeshComponent.mIndexCount );
        //     } );

        // mAccelerationStructure->Build();
    }

    void Scene::InitializeRayTracing()
    {
        mRayTracingContext     = SE::Core::New<OptixDeviceContextObject>();
        mAccelerationStructure = SE::Core::New<OptixScene>( mRayTracingContext );
    }

    void Scene::Render()
    {
    }

    static void WriteNode( ConfigurationWriter &lOut, Entity const &aEntity, sUUID const &aUUID,
                           std::vector<sImportedAnimationSampler>       &lInterpolationData,
                           std::unordered_map<std::string, std::string> &aMaterialMap,
                           std::unordered_map<std::string, std::string> &aMeshDataMap )
    {
        lOut.WriteKey( aUUID.mValue.str() );
        lOut.BeginMap();
        {
            if( aEntity.Has<sTag>() )
            {
                WriteComponent( lOut, aEntity.Get<sTag>() );
            }
            if( aEntity.Has<sRelationshipComponent>() )
                WriteComponent( lOut, aEntity.Get<sRelationshipComponent>() );
            if( aEntity.Has<sCameraComponent>() )
                WriteComponent( lOut, aEntity.Get<sCameraComponent>() );
            if( aEntity.Has<sAnimationChooser>() )
                WriteComponent( lOut, aEntity.Get<sAnimationChooser>() );
            if( aEntity.Has<sActorComponent>() )
                WriteComponent( lOut, aEntity.Get<sActorComponent>() );

            if( aEntity.Has<sAnimationComponent>() )
            {
                auto &lComponent = aEntity.Get<sAnimationComponent>();
                lOut.WriteKey( TypeTag<sAnimationComponent>() );
                lOut.BeginMap();
                {
                    lOut.WriteKey( "Duration", lComponent.Duration );
                    lOut.WriteKey( "mChannels" );
                    lOut.BeginSequence();
                    {
                        for( auto &lAnimationChannel : lComponent.mChannels )
                        {
                            lOut.BeginMap( true );
                            {
                                lOut.WriteKey( "mTargetNode", lAnimationChannel.mTargetNode.Get<sUUID>().mValue.str() );
                                lOut.WriteKey( "mChannelID", (uint32_t)lAnimationChannel.mChannelID );
                                lOut.WriteKey( "mInterpolationDataIndex", lInterpolationData.size() );
                                lInterpolationData.push_back( lAnimationChannel.mInterpolation );
                            }
                            lOut.EndMap();
                        }
                    }
                    lOut.EndSequence();
                }
                lOut.EndMap();
            }

            if( aEntity.Has<sAnimatedTransformComponent>() )
                WriteComponent( lOut, aEntity.Get<sAnimatedTransformComponent>() );

            if( aEntity.Has<sNodeTransformComponent>() )
                WriteComponent( lOut, aEntity.Get<sNodeTransformComponent>() );

            if( aEntity.Has<sStaticMeshComponent>() )
                WriteComponent( lOut, aEntity.Get<sStaticMeshComponent>(), aMeshDataMap[aUUID.mValue.str()] );

            if( aEntity.Has<sParticleSystemComponent>() )
                WriteComponent( lOut, aEntity.Get<sParticleSystemComponent>() );

            if( aEntity.Has<sParticleShaderComponent>() )
                WriteComponent( lOut, aEntity.Get<sParticleShaderComponent>() );

            if( aEntity.Has<sSkeletonComponent>() )
                WriteComponent( lOut, aEntity.Get<sSkeletonComponent>() );

            if( aEntity.Has<sWireframeComponent>() )
                WriteComponent( lOut, aEntity.Get<sWireframeComponent>() );

            if( aEntity.Has<sWireframeMeshComponent>() )
                WriteComponent( lOut, aEntity.Get<sWireframeMeshComponent>() );

            if( aEntity.Has<sBoundingBoxComponent>() )
                WriteComponent( lOut, aEntity.Get<sBoundingBoxComponent>() );

            if( aEntity.Has<sRayTracingTargetComponent>() )
                WriteComponent( lOut, aEntity.Get<sRayTracingTargetComponent>() );

            if( aEntity.Has<sMaterialComponent>() )
                WriteComponent( lOut, aEntity.Get<sMaterialComponent>(), aMaterialMap[aUUID.mValue.str()] );

            if( aEntity.Has<sMaterialShaderComponent>() )
                WriteComponent( lOut, aEntity.Get<sMaterialShaderComponent>() );

            if( aEntity.Has<sBackgroundComponent>() )
                WriteComponent( lOut, aEntity.Get<sBackgroundComponent>() );

            if( aEntity.Has<sAmbientLightingComponent>() )
                WriteComponent( lOut, aEntity.Get<sAmbientLightingComponent>() );

            if( aEntity.Has<sLightComponent>() )
                WriteComponent( lOut, aEntity.Get<sLightComponent>() );

            if( aEntity.Has<sUIComponent>() )
                WriteComponent( lOut, aEntity.Get<sUIComponent>() );
        }
        lOut.EndMap();
    }

    void Scene::SaveAs( fs::path aPath )
    {
        // Check that path does not exist, or exists and is a folder
        if( !fs::exists( aPath ) )
            fs::create_directories( aPath );
        if( !fs::is_directory( aPath ) )
            return;

        if( !fs::exists( aPath / "Materials" ) )
            fs::create_directories( aPath / "Materials" );
        if( !fs::exists( aPath / "Meshes" ) )
            fs::create_directories( aPath / "Meshes" );
        if( !fs::exists( aPath / "Animations" ) )
            fs::create_directories( aPath / "Animations" );

        std::vector<std::string> lMaterialList{};
        {
            auto &lMaterials = mMaterialSystem->GetMaterialData();
            auto &lTextures  = mMaterialSystem->GetTextures();

            for( auto &lMaterial : lMaterials )
            {
                BinaryAsset lBinaryDataFile{};
                std::string lMaterialFileName = fmt::format( "{}.material", lMaterial.mName );

                auto lRetrieveAndPackageTexture = [&]( sTextureReference aTexture )
                {
                    if( aTexture.mTextureID >= 2 )
                    {
                        sTextureSamplingInfo lSamplingInfo = lTextures[aTexture.mTextureID]->mSpec;
                        TextureData2D        lTextureData;
                        lTextures[aTexture.mTextureID]->GetTexture()->GetPixelData( lTextureData );
                        lBinaryDataFile.Package( lTextureData, lSamplingInfo );
                    }
                };

                uint32_t          lCurrentTextureID = 0;
                sMaterial         lNewMaterial      = lMaterial;
                sTextureReference lDefaultTexture{ 0, std::numeric_limits<uint32_t>::max() };

                lNewMaterial.mBaseColorTexture = lDefaultTexture;
                if( lMaterial.mBaseColorTexture.mTextureID >= 2 )
                    lNewMaterial.mBaseColorTexture = sTextureReference{ 0, lCurrentTextureID++ };

                lNewMaterial.mNormalsTexture = lDefaultTexture;
                if( lMaterial.mNormalsTexture.mTextureID >= 2 )
                    lNewMaterial.mNormalsTexture = sTextureReference{ 0, lCurrentTextureID++ };

                lNewMaterial.mMetalRoughTexture = lDefaultTexture;
                if( lMaterial.mMetalRoughTexture.mTextureID >= 2 )
                    lNewMaterial.mMetalRoughTexture = sTextureReference{ 0, lCurrentTextureID++ };

                lNewMaterial.mOcclusionTexture = lDefaultTexture;
                if( lMaterial.mOcclusionTexture.mTextureID >= 2 )
                    lNewMaterial.mOcclusionTexture = sTextureReference{ 0, lCurrentTextureID++ };

                lNewMaterial.mEmissiveTexture = lDefaultTexture;
                if( lMaterial.mEmissiveTexture.mTextureID >= 2 )
                    lNewMaterial.mEmissiveTexture = sTextureReference{ 0, lCurrentTextureID++ };

                lBinaryDataFile.Package( lNewMaterial );

                lRetrieveAndPackageTexture( lMaterial.mBaseColorTexture );
                lRetrieveAndPackageTexture( lMaterial.mNormalsTexture );
                lRetrieveAndPackageTexture( lMaterial.mMetalRoughTexture );
                lRetrieveAndPackageTexture( lMaterial.mOcclusionTexture );
                lRetrieveAndPackageTexture( lMaterial.mEmissiveTexture );

                lBinaryDataFile.WriteTo( aPath / "Materials" / lMaterialFileName );
                lMaterialList.push_back( fmt::format( "{}/{}", "Materials", lMaterialFileName ) );
            }
        }
        std::unordered_map<std::string, std::string> lMaterialMap{};
        ForEach<sMaterialComponent>( [&]( auto aEntity, auto &aComponent )
                                     { lMaterialMap[aEntity.Get<sUUID>().mValue.str()] = lMaterialList[aComponent.mMaterialID]; } );

        std::unordered_map<std::string, std::string> lMeshPathDB{};
        ForEach<sStaticMeshComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                BinaryAsset lBinaryDataFile{};
                std::string lSerializedMeshName = fmt::format( "{}.mesh", aEntity.Get<sUUID>().mValue.str() );

                auto lVertexData = aComponent.mVertexBuffer->Fetch<VertexData>();
                auto lIndexData  = aComponent.mIndexBuffer->Fetch<uint32_t>();
                lBinaryDataFile.Package( lVertexData, lIndexData );
                lBinaryDataFile.WriteTo( aPath / "Meshes" / lSerializedMeshName );
                lMeshPathDB[aEntity.Get<sUUID>().mValue.str()] = fmt::format( "{}/{}", "Meshes", lSerializedMeshName );
            } );

        auto lOut = ConfigurationWriter( aPath / "SceneDefinition.yaml" );

        std::vector<sImportedAnimationSampler> lInterpolationData;
        lOut.BeginMap();
        lOut.WriteKey( "scene" );
        {
            lOut.BeginMap();
            lOut.WriteKey( "name", "FOO" );
            lOut.WriteKey( "version", "1" );
            lOut.WriteKey( "root", Root.Get<sUUID>().mValue.str() );
            lOut.WriteKey( "environment", Environment.Get<sUUID>().mValue.str() );
            lOut.WriteKey( "default_camera", DefaultCamera.Get<sUUID>().mValue.str() );
            lOut.WriteKey( "current_camera", CurrentCamera.Get<sUUID>().mValue.str() );
            lOut.WriteKey( "editor_view" );
            lOut.Write( mEditorView );
            lOut.WriteKey( "nodes" );
            {
                lOut.BeginMap();
                ForEach<sUUID>( [&]( auto aEntity, auto &aUUID )
                                { WriteNode( lOut, aEntity, aUUID, lInterpolationData, lMaterialMap, lMeshPathDB ); } );
                lOut.EndMap();
            }
            lOut.EndMap();
        }
        lOut.EndMap();

        fs::path    lOutput = aPath / "Animations" / "BinaryData.bin";
        BinaryAsset lBinaryDataFile;
        for( auto &lInterpolation : lInterpolationData )
            lBinaryDataFile.Package( lInterpolation );
        lBinaryDataFile.WriteTo( lOutput );
    }
} // namespace SE::Core
