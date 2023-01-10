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

    Scene::Scene( Ref<VkGraphicContext> a_GraphicContext, Ref<SE::Core::UIContext> a_UI )
        : mGraphicContext{ a_GraphicContext }
    {
        mMaterialSystem = New<MaterialSystem>( a_GraphicContext );

        DefaultCamera                 = mRegistry.CreateEntity( "DefaultCamera" );
        auto &l_CameraComponent       = DefaultCamera.Add<sCameraComponent>();
        l_CameraComponent.Position    = math::vec3{ 0.0f, 0.0f, 0.0f };
        l_CameraComponent.Pitch       = 0.0f;
        l_CameraComponent.Yaw         = 0.0f;
        l_CameraComponent.Roll        = 0.0f;
        l_CameraComponent.Near        = 0.001;
        l_CameraComponent.Far         = 1000.0f;
        l_CameraComponent.FieldOfView = 90.0f;
        l_CameraComponent.AspectRatio = 16.0f / 9.0f;

        CurrentCamera = DefaultCamera;

        Environment = mRegistry.CreateEntity( "Environment" );
        Environment.Add<sAmbientLightingComponent>();
        Environment.Add<sBackgroundComponent>();

        Root = mRegistry.CreateEntityWithRelationship( "WorldRoot" );
        Root.Add<sNodeTransformComponent>();

        InitializeRayTracing();
    }

    template <typename _Component>
    static void CopyComponent( Entity &aSource, Entity &aDestination )
    {
        if( ( aSource.Has<_Component>() ) ) aDestination.AddOrReplace<_Component>( aSource.Get<_Component>() );
    }

    Scene::Scene( Ref<Scene> aSource )
    {
        mGraphicContext = aSource->mGraphicContext;
        mMaterialSystem = aSource->mMaterialSystem;

        InitializeRayTracing();

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
            CopyComponent<sTransformMatrixComponent>( lEntity, lClonedEntity );
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

        uint32_t lTransformCount = 0;
        aSource->ForEach<sNodeTransformComponent>( [&]( auto aEntity, auto &aUUID ) { lTransformCount++; } );

        uint32_t lStaticMeshCount = 0;
        aSource->ForEach<sStaticMeshComponent>( [&]( auto aEntity, auto &aUUID ) { lStaticMeshCount++; } );

        uint32_t lJointMatrixCount = 0;
        uint32_t lJointOffsetCount = 0;
        ForEach<sSkeletonComponent>(
            [&]( auto lElementToProcess, auto &s )
            {
                lJointMatrixCount += s.JointMatrices.size();
                lJointOffsetCount += 1;
            } );

        mTransforms         = GPUMemory::Create<math::mat4>( lTransformCount );
        mVertexBuffers      = GPUMemory::Create<VkGpuBuffer>( lStaticMeshCount );
        mTransformedBuffers = GPUMemory::Create<VkGpuBuffer>( lStaticMeshCount );
        mVertexOffsets      = GPUMemory::Create<uint32_t>( lStaticMeshCount );
        mVertexCounts       = GPUMemory::Create<uint32_t>( lStaticMeshCount );
        mJointTransforms    = GPUMemory::Create<math::mat4>( lJointMatrixCount );
        mJointOffsets       = GPUMemory::Create<uint32_t>( lJointOffsetCount );

        mIsClone = true;
    }

    Scene::~Scene()
    {
        mTransforms.Dispose();
        mVertexOffsets.Dispose();
        mVertexCounts.Dispose();
    }

    math::mat4 Scene::GetView()
    {
        math::mat4 l_CameraView( 1.0f );
        if( CurrentCamera.Has<sCameraComponent>() )
        {
            auto &lComponent = CurrentCamera.Get<sCameraComponent>();

            math::mat4 lRx = math::Rotation( lComponent.Pitch, math::vec3( 1.0f, 0.0f, 0.0f ) );
            math::mat4 lRy = math::Rotation( lComponent.Yaw, math::vec3( 0.0f, 1.0f, 0.0f ) );
            math::mat4 lRz = math::Rotation( -lComponent.Roll, math::vec3( 0.0f, 0.0f, 1.0f ) );

            l_CameraView = math::Inverse( math::Translate( lRx * lRy * lRz, lComponent.Position ) );
        }

        return l_CameraView;
    }

    math::mat4 Scene::GetProjection()
    {
        math::mat4 l_CameraProjection( 1.0f );
        if( CurrentCamera.Has<sCameraComponent>() )
        {
            auto &lComponent = CurrentCamera.Get<sCameraComponent>();
            l_CameraProjection =
                math::Perspective( math::radians( lComponent.FieldOfView ), lComponent.AspectRatio, lComponent.Near, lComponent.Far );
            l_CameraProjection[1][1] *= -1.0f;
        }
        return l_CameraProjection;
    }

    math::vec3 Scene::GetCameraPosition()
    {
        math::vec3 l_CameraPosition( 0.0f );
        if( CurrentCamera.Has<sCameraComponent>() )
        {
            auto &lComponent = CurrentCamera.Get<sCameraComponent>();
            l_CameraPosition = lComponent.Position;
        }
        return l_CameraPosition;
    }

    Scene::Element Scene::Create( std::string a_Name, Element a_Parent ) { return mRegistry.CreateEntity( a_Parent, a_Name ); }

    Scene::Element Scene::CreateEntity() { return mRegistry.CreateEntity(); }

    Scene::Element Scene::CreateEntity( std::string a_Name ) { return mRegistry.CreateEntity( a_Name ); }

    void Scene::ClearScene()
    {
        mMaterialSystem->Wipe();
        mRegistry.Clear();

        mMaterialSystem->Clear();

        DefaultCamera                 = mRegistry.CreateEntity( "DefaultCamera" );
        auto &l_CameraComponent       = DefaultCamera.Add<sCameraComponent>();
        l_CameraComponent.Position    = math::vec3{ 0.0f, 0.0f, 0.0f };
        l_CameraComponent.Pitch       = 0.0f;
        l_CameraComponent.Yaw         = 0.0f;
        l_CameraComponent.Roll        = 0.0f;
        l_CameraComponent.Near        = 0.001;
        l_CameraComponent.Far         = 1000.0f;
        l_CameraComponent.FieldOfView = 90.0f;
        l_CameraComponent.AspectRatio = 16.0f / 9.0f;

        CurrentCamera = DefaultCamera;

        Environment = mRegistry.CreateEntity( "Environment" );
        Environment.Add<sAmbientLightingComponent>();
        Environment.Add<sBackgroundComponent>();

        Root = mRegistry.CreateEntityWithRelationship( "WorldRoot" );
        Root.Add<sNodeTransformComponent>();
    }

    Scene::Element Scene::LoadModel( Ref<sImportedModel> aModelData, math::mat4 aTransform )
    {
        return LoadModel( aModelData, aTransform, "MODEL" );
    }

    template <typename _Ty>
    void ReadComponent( Entity aEntity, ConfigurationNode const &aNode, std::unordered_map<std::string, Entity> &aEntities )
    {
        //
    }

    template <typename _Ty>
    void ReadComponent( Entity aEntity, ConfigurationNode const &aNode, std::unordered_map<std::string, Entity> &aEntities,
                        std::vector<sImportedAnimationSampler> &aInterpolationData )
    {
        //
    }

    template <>
    void ReadComponent<sTag>( Entity aEntity, ConfigurationNode const &aNode, std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sTag"].IsNull() )
        {
            aEntity.Add<sTag>( aNode["sTag"]["mValue"].As<std::string>( "" ) );
        }
    }

    template <>
    void ReadComponent<sCameraComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                          std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sCameraComponent"].IsNull() )
        {
            auto &lComponent       = aEntity.Add<sCameraComponent>();
            lComponent.Position    = aNode["sCameraComponent"]["Position"].Vec( { "x", "y", "z" }, math::vec3{ 0.0f, 0.0f, 0.0f } );
            lComponent.Pitch       = aNode["sCameraComponent"]["Pitch"].As<float>( 0.0f );
            lComponent.Yaw         = aNode["sCameraComponent"]["Yaw"].As<float>( 0.0f );
            lComponent.Roll        = aNode["sCameraComponent"]["Roll"].As<float>( 0.0f );
            lComponent.Near        = aNode["sCameraComponent"]["Near"].As<float>( 0.0f );
            lComponent.Far         = aNode["sCameraComponent"]["Far"].As<float>( 0.0f );
            lComponent.FieldOfView = aNode["sCameraComponent"]["FieldOfView"].As<float>( 0.0f );
            lComponent.AspectRatio = aNode["sCameraComponent"]["AspectRatio"].As<float>( 0.0f );
        }
    }

    template <>
    void ReadComponent<sActorComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                         std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sActorComponent"].IsNull() )
        {
            auto  lClassFullName      = aNode["sActorComponent"]["mClassFullName"].As<std::string>( "" );
            auto &lNewScriptComponent = aEntity.Add<sActorComponent>( lClassFullName );

            lNewScriptComponent.Initialize( aEntity );
        }
    }

    template <>
    void ReadComponent<sAnimationChooser>( Entity aEntity, ConfigurationNode const &aNode,
                                           std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sAnimationChooser"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sAnimationChooser>();
            aNode["sAnimationChooser"].ForEach(
                [&]( ConfigurationNode &aNode )
                {
                    std::string lAnimationUUID = aNode.As<std::string>( "" );
                    Entity      lAnimationNode = aEntities[lAnimationUUID];

                    lComponent.Animations.push_back( lAnimationNode );
                    SE::Logging::Info( "ANIMATION {}", lAnimationUUID );
                } );
        }
    }

    template <>
    void ReadComponent<sAnimationComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                             std::unordered_map<std::string, Entity> &aEntities,
                                             std::vector<sImportedAnimationSampler>  &aInterpolationData )
    {
        if( !aNode["sAnimationComponent"].IsNull() )
        {
            auto &lComponent     = aEntity.Add<sAnimationComponent>();
            lComponent.Duration  = aNode["sAnimationComponent"]["Duration"].As<float>( 0.0f );
            lComponent.mChannels = std::vector<sAnimationChannel>{};

            aNode["sAnimationComponent"]["mChannels"].ForEach(
                [&]( ConfigurationNode &aInterpolationDataNode )
                {
                    sAnimationChannel lNewChannel{};
                    std::string       lTargetNodeUUID = aInterpolationDataNode["mTargetNode"].As<std::string>( "" );

                    lNewChannel.mTargetNode = aEntities[lTargetNodeUUID];
                    lNewChannel.mChannelID =
                        static_cast<sImportedAnimationChannel::Channel>( aInterpolationDataNode["mChannelID"].As<uint32_t>( 0 ) );
                    lNewChannel.mInterpolation =
                        aInterpolationData[aInterpolationDataNode["mInterpolationDataIndex"].As<uint32_t>( 0 )];

                    lComponent.mChannels.push_back( lNewChannel );
                } );
        }
    }

    template <>
    void ReadComponent<sAnimatedTransformComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                                     std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sAnimatedTransformComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sAnimatedTransformComponent>();

            lComponent.Translation =
                aNode["sAnimatedTransformComponent"]["Translation"].Vec( { "x", "y", "z" }, math::vec3{ 0.0f, 0.0f, 0.0f } );
            lComponent.Scaling =
                aNode["sAnimatedTransformComponent"]["Scaling"].Vec( { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );

            auto lCoefficients =
                aNode["sAnimatedTransformComponent"]["Rotation"].Vec( { "x", "y", "z", "w" }, math::vec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
            lComponent.Rotation.x = lCoefficients.x;
            lComponent.Rotation.y = lCoefficients.y;
            lComponent.Rotation.z = lCoefficients.z;
            lComponent.Rotation.w = lCoefficients.w;
        }
    }

    math::mat4 ReadMatrix( ConfigurationNode &aNode )
    {

        std::vector<float> lMatrixEntries{};
        aNode.ForEach( [&]( ConfigurationNode &aNode ) { lMatrixEntries.push_back( aNode.As<float>( 0.0f ) ); } );

        math::mat4 lMatrix;
        for( uint32_t c = 0; c < 4; c++ )
            for( uint32_t r = 0; r < 4; r++ ) lMatrix[c][r] = lMatrixEntries[4 * c + r];

        return lMatrix;
    }

    void ReadMatrix( math::mat4 &aMatrix, ConfigurationNode &aNode )
    {

        std::vector<float> lMatrixEntries{};
        aNode.ForEach( [&]( ConfigurationNode &aNode ) { lMatrixEntries.push_back( aNode.As<float>( 0.0f ) ); } );

        for( uint32_t c = 0; c < 4; c++ )
            for( uint32_t r = 0; r < 4; r++ ) aMatrix[c][r] = lMatrixEntries[4 * c + r];
    }

    template <>
    void ReadComponent<sNodeTransformComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                                 std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sLocalTransformComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sNodeTransformComponent>();

            ReadMatrix( lComponent.mMatrix, aNode["sLocalTransformComponent"]["mMatrix"] );
        }
    }

    template <>
    void ReadComponent<sTransformMatrixComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                                   std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["TransformMatrixComponent"].IsNull() )
        {
            auto &lComponent = aEntity.AddOrReplace<sTransformMatrixComponent>();

            ReadMatrix( lComponent.Matrix, aNode["TransformMatrixComponent"]["mMatrix"] );
        }
    }

    template <>
    void ReadComponent<sStaticMeshComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                              std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sStaticMeshComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sStaticMeshComponent>();

            auto &lMeshData          = aNode["sStaticMeshComponent"];
            lComponent.mVertexOffset = lMeshData["mVertexOffset"].As<uint32_t>( 0 );
            lComponent.mVertexCount  = lMeshData["mVertexCount"].As<uint32_t>( 0 );
            lComponent.mIndexOffset  = lMeshData["mIndexOffset"].As<uint32_t>( 0 );
            lComponent.mIndexCount   = lMeshData["mIndexCount"].As<uint32_t>( 0 );
        }
    }

    template <>
    void ReadComponent<sParticleSystemComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                                  std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sParticleSystemComponent"].IsNull() )
        {
        }
    }

    template <>
    void ReadComponent<sParticleShaderComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                                  std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sParticleShaderComponent"].IsNull() )
        {
        }
    }

    template <>
    void ReadComponent<sWireframeComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                             std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sWireframeComponent"].IsNull() )
        {
        }
    }

    template <>
    void ReadComponent<sWireframeMeshComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                                 std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sWireframeMeshComponent"].IsNull() )
        {
        }
    }

    template <>
    void ReadComponent<sBoundingBoxComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                               std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sBoundingBoxComponent"].IsNull() )
        {
        }
    }

    template <>
    void ReadComponent<sSkeletonComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                            std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sSkeletonComponent"].IsNull() )
        {
            auto &lData = aNode["sSkeletonComponent"];

            std::vector<Entity> lBones{};
            lData["Bones"].ForEach(
                [&]( ConfigurationNode &aNode )
                {
                    auto lUUID = aNode.As<std::string>( "" );
                    if( lUUID.empty() ) return;

                    lBones.push_back( aEntities[lUUID] );
                } );

            std::vector<math::mat4> lInverseBindMatrices{};
            lData["InverseBindMatrices"].ForEach( [&]( ConfigurationNode &aNode )
                                                  { lInverseBindMatrices.push_back( ReadMatrix( aNode ) ); } );

            std::vector<math::mat4> lJointMatrices{};
            lData["JointMatrices"].ForEach( [&]( ConfigurationNode &aNode ) { lJointMatrices.push_back( ReadMatrix( aNode ) ); } );

            auto &lComponent               = aEntity.Add<sSkeletonComponent>();
            lComponent.BoneCount           = lBones.size();
            lComponent.Bones               = lBones;
            lComponent.InverseBindMatrices = lInverseBindMatrices;
            lComponent.JointMatrices       = lJointMatrices;
        }
    }

    template <>
    void ReadComponent<sRayTracingTargetComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                                    std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sRayTracingTargetComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sRayTracingTargetComponent>();

            ReadMatrix( lComponent.Transform, aNode["sRayTracingTargetComponent"]["Transform"] );
        }
    }

    template <>
    void ReadComponent<sMaterialComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                            std::unordered_map<std::string, Entity> &aEntities )
    {

        if( !aNode["sMaterialComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sMaterialComponent>();

            lComponent.mMaterialID = aNode["sMaterialComponent"]["mMaterialID"].As<uint32_t>( 0 );
            SE::Logging::Info( "{}", lComponent.mMaterialID );
        }
    }

    template <>
    void ReadComponent<sMaterialShaderComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                                  std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sMaterialShaderComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sMaterialShaderComponent>();
            auto &lData      = aNode["sMaterialShaderComponent"];

            lComponent.Type              = static_cast<eCMaterialType>( lData["Type"].As<uint8_t>( 0 ) );
            lComponent.IsTwoSided        = lData["IsTwoSided"].As<bool>( true );
            lComponent.UseAlphaMask      = lData["UseAlphaMask"].As<bool>( true );
            lComponent.LineWidth         = lData["LineWidth"].As<float>( 1.0f );
            lComponent.AlphaMaskTheshold = lData["AlphaMaskTheshold"].As<float>( .5f );
        }
    }

    template <>
    void ReadComponent<sBackgroundComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                              std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sBackgroundComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sBackgroundComponent>();

            lComponent.Color = aNode["sBackgroundComponent"]["Color"].Vec( { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );
        }
    }

    template <>
    void ReadComponent<sAmbientLightingComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                                   std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sAmbientLightingComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sAmbientLightingComponent>();

            lComponent.Color = aNode["sAmbientLightingComponent"]["Color"].Vec( { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );
            lComponent.Intensity = aNode["sAmbientLightingComponent"]["Intensity"].As<float>( .0005f );
        }
    }

    template <>
    void ReadComponent<sLightComponent>( Entity aEntity, ConfigurationNode const &aNode,
                                         std::unordered_map<std::string, Entity> &aEntities )
    {
        if( !aNode["sLightComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sLightComponent>();

            std::unordered_map<std::string, eLightType> lLightTypeLookup = { { "DIRECTIONAL", eLightType::DIRECTIONAL },
                                                                             { "SPOTLIGHT", eLightType::SPOTLIGHT },
                                                                             { "POINT_LIGHT", eLightType::POINT_LIGHT },
                                                                             { "", eLightType::POINT_LIGHT } };

            lComponent.mType      = lLightTypeLookup[aNode["sLightComponent"]["mType"].As<std::string>( "" )];
            lComponent.mColor     = aNode["sLightComponent"]["mColor"].Vec( { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );
            lComponent.mIntensity = aNode["sLightComponent"]["mIntensity"].As<float>( .0005f );
            lComponent.mCone      = aNode["sLightComponent"]["mCone"].As<float>( .0005f );
        }
    }

    void Scene::LoadScenario( fs::path aScenarioPath )
    {
        mRegistry.Clear();
        mMaterialSystem->Wipe();

        auto lScenarioRoot = aScenarioPath.parent_path();
        auto lScenarioData = BinaryAsset( lScenarioRoot / "BinaryData.bin" );

        auto lOffseIndex = lScenarioData.GetIndex( 0 );
        if( lOffseIndex.mType != eAssetType::OFFSET_DATA ) throw std::runtime_error( "Binary data type mismatch" );
        lScenarioData.Seek( lOffseIndex.mByteStart );
        auto lMaterialOffset  = lScenarioData.Read<uint32_t>();
        auto lMaterialCount   = lScenarioData.Read<uint32_t>();
        auto lTextureOffset   = lScenarioData.Read<uint32_t>();
        auto lTextureCount    = lScenarioData.Read<uint32_t>();
        auto lAnimationOffset = lScenarioData.Read<uint32_t>();
        auto lAnimationCount  = lScenarioData.Read<uint32_t>();

        std::vector<VertexData> lVertexBuffer;
        std::vector<uint32_t>   lIndexBuffer;
        lScenarioData.Retrieve( 1, lVertexBuffer, lIndexBuffer );

        for( uint32_t lMaterialIndex = 0; lMaterialIndex < lMaterialCount; lMaterialIndex++ )
        {
            sMaterial lMaterialData;
            lScenarioData.Retrieve( lMaterialIndex + lMaterialOffset, lMaterialData );

            auto &lNewMaterial = mMaterialSystem->CreateMaterial( lMaterialData );
        }

        for( uint32_t lTextureIndex = 0; lTextureIndex < lTextureCount; lTextureIndex++ )
        {
            auto [aData, aSampler] = lScenarioData.Retrieve( lTextureIndex + lTextureOffset );

            auto lNewTexture = mMaterialSystem->CreateTexture( aData, aSampler );
        }

        std::vector<sImportedAnimationSampler> lInterpolationData;
        for( uint32_t lInterpolationIndex = 0; lInterpolationIndex < lAnimationCount; lInterpolationIndex++ )
        {
            auto &lAnimationData = lInterpolationData.emplace_back();

            lScenarioData.Retrieve( lInterpolationIndex + lAnimationOffset, lAnimationData );
        }

        auto lScenarioDescription = ConfigurationReader( aScenarioPath );

        std::unordered_map<std::string, Entity>      lEntities{};
        std::unordered_map<std::string, std::string> lParentEntityLUT{};

        auto &lSceneRoot = lScenarioDescription.GetRoot()["scene"];

        lSceneRoot["nodes"].ForEach<std::string>(
            [&]( auto const &aKey, auto const &aValue )
            {
                auto lEntity = mRegistry.CreateEntity( sUUID( aKey ) );
                auto lUUID   = lEntity.Get<sUUID>().mValue;

                lEntities[aKey] = lEntity;

                if( !aValue["sRelationshipComponent"].IsNull() )
                {
                    if( !( aValue["sRelationshipComponent"]["mParent"].IsNull() ) )
                    {
                        auto lParentUUIDStr    = aValue["sRelationshipComponent"]["mParent"].As<std::string>( "" );
                        auto lParentUUID       = UUIDv4::UUID::fromStrFactory( lParentUUIDStr );
                        lParentEntityLUT[aKey] = lParentUUIDStr;
                    }
                }
            } );

        lSceneRoot["nodes"].ForEach<std::string>(
            [&]( auto const &aKey, auto const &lEntityConfiguration )
            {
                auto  lUUID   = UUIDv4::UUID::fromStrFactory( aKey );
                auto &lEntity = lEntities[aKey];

                if( !lEntity ) return;

                if( lParentEntityLUT.find( aKey ) != lParentEntityLUT.end() )
                    mRegistry.SetParent( lEntity, lEntities[lParentEntityLUT[aKey]] );

                ReadComponent<sTag>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sCameraComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sAnimationChooser>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sAnimationComponent>( lEntity, lEntityConfiguration, lEntities, lInterpolationData );
                ReadComponent<sAnimatedTransformComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sNodeTransformComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sTransformMatrixComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sStaticMeshComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sSkeletonComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sRayTracingTargetComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sMaterialComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sMaterialShaderComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sBackgroundComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sAmbientLightingComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sLightComponent>( lEntity, lEntityConfiguration, lEntities );
                ReadComponent<sActorComponent>( lEntity, lEntityConfiguration, lEntities );

                SE::Logging::Info( "Components added to entity {}", aKey );
            } );

        auto lRootNodeUUIDStr = lSceneRoot["root"].As<std::string>( "" );
        auto lRootNodeUUID    = UUIDv4::UUID::fromStrFactory( lRootNodeUUIDStr );
        Root                  = lEntities[lRootNodeUUIDStr];
        SE::Logging::Info( "Created root", lRootNodeUUIDStr );

        auto lEnvironmentNodeUUID = lSceneRoot["environment"].As<std::string>( "" );
        Environment               = lEntities[lEnvironmentNodeUUID];
        SE::Logging::Info( "Created environment", lEnvironmentNodeUUID );

        auto lCurrentCameraUUID = lSceneRoot["current_camera"].As<std::string>( "" );
        CurrentCamera           = lEntities[lCurrentCameraUUID];
        SE::Logging::Info( "Created camera", lCurrentCameraUUID );

        auto lDefaultCameraUUID = lSceneRoot["default_camera"].As<std::string>( "" );
        DefaultCamera           = lEntities[lDefaultCameraUUID];
        SE::Logging::Info( "Created camera", lDefaultCameraUUID );

        uint32_t lTransformCount = 0;
        ForEach<sNodeTransformComponent>( [&]( auto aEntity, auto &aUUID ) { lTransformCount++; } );

        uint32_t lStaticMeshCount = 0;
        ForEach<sStaticMeshComponent>( [&]( auto aEntity, auto &aUUID ) { lStaticMeshCount++; } );

        uint32_t lJointMatrixCount = 0;
        uint32_t lJointOffsetCount = 0;
        ForEach<sSkeletonComponent>(
            [&]( auto lElementToProcess, auto &s )
            {
                lJointMatrixCount += s.JointMatrices.size();
                lJointOffsetCount += 1;
            } );

        mTransforms         = GPUMemory::Create<math::mat4>( static_cast<uint32_t>( lTransformCount ) );
        mVertexBuffers      = GPUMemory::Create<VkGpuBuffer>( lStaticMeshCount );
        mTransformedBuffers = GPUMemory::Create<VkGpuBuffer>( lStaticMeshCount );
        mVertexOffsets      = GPUMemory::Create<uint32_t>( static_cast<uint32_t>( lStaticMeshCount ) );
        mVertexCounts       = GPUMemory::Create<uint32_t>( static_cast<uint32_t>( lStaticMeshCount ) );
        mJointTransforms    = GPUMemory::Create<math::mat4>( lJointMatrixCount );
        mJointOffsets       = GPUMemory::Create<uint32_t>( lJointOffsetCount );
    }

    Scene::Element Scene::LoadModel( Ref<sImportedModel> aModelData, math::mat4 aTransform, std::string a_Name )
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
            lMaterialShader.Type              = eCMaterialType::Opaque;
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
                lVertices[i].TexCoords_1 = lMesh.mUV1[i];
                lVertices[i].Bones       = lMesh.mJoints[i];
                lVertices[i].Weights     = lMesh.mWeights[i];
            }

            lMeshComponent.mVertexBuffer =
                New<VkGpuBuffer>( mGraphicContext, lVertices, eBufferType::VERTEX_BUFFER, false, false, true, true );
            lMeshComponent.mIndexBuffer =
                New<VkGpuBuffer>( mGraphicContext, lMesh.mIndices, eBufferType::INDEX_BUFFER, false, false, true, true );
            lMeshComponent.mTransformedBuffer = New<VkGpuBuffer>( mGraphicContext, eBufferType::VERTEX_BUFFER, false, false, true,
                                                                  true, lMeshComponent.mVertexBuffer->SizeAs<uint8_t>() );

            lMeshComponent.mVertexOffset = 0;
            lMeshComponent.mVertexCount  = lVertices.size();
            lMeshComponent.mIndexOffset  = 0;
            lMeshComponent.mIndexCount   = lMesh.mIndices.size();

            auto lMeshEntity = Create( lMesh.mName, lAssetEntity );
            lMeshEntity.Add<sStaticMeshComponent>( lMeshComponent );
            lMeshEntity.Add<sMaterialComponent>( lMaterialIds[lMesh.mMaterialID] );
            lMeshEntity.Add<sMaterialShaderComponent>( lMaterialCreateInfo[lMesh.mMaterialID] );
            lMeshEntity.Add<sNodeTransformComponent>( math::mat4( 1.0f ) );

            lMeshes.push_back( lMeshEntity );
        }

        uint32_t lTransformCount = 0;
        ForEach<sNodeTransformComponent>( [&]( auto aEntity, auto &aUUID ) { lTransformCount++; } );

        uint32_t lStaticMeshCount = 0;
        ForEach<sStaticMeshComponent>( [&]( auto aEntity, auto &aUUID ) { lStaticMeshCount++; } );

        mTransforms         = GPUMemory::Create<math::mat4>( lTransformCount );
        mVertexBuffers      = GPUMemory::Create<VkGpuBuffer>( lStaticMeshCount );
        mTransformedBuffers = GPUMemory::Create<VkGpuBuffer>( lStaticMeshCount );
        mVertexOffsets      = GPUMemory::Create<uint32_t>( lStaticMeshCount );
        mVertexCounts       = GPUMemory::Create<uint32_t>( lStaticMeshCount );

        std::vector<Element> lNodes = {};
        for( auto &lNode : aModelData->mNodes )
        {
            auto l_NodeEntity = mRegistry.CreateEntityWithRelationship( lNode.mName );
            l_NodeEntity.Add<sNodeTransformComponent>( lNode.mTransform );

            lNodes.push_back( l_NodeEntity );
        }

        int32_t l_Index = 0;
        for( auto &l_NodeEntity : lNodes )
        {
            if( aModelData->mNodes[l_Index].mParentID == std::numeric_limits<uint32_t>::max() )
                mRegistry.SetParent( l_NodeEntity, lAssetEntity );
            else
                mRegistry.SetParent( l_NodeEntity, lNodes[aModelData->mNodes[l_Index].mParentID] );
            l_Index++;
        }

        for( uint32_t lNodeID = 0; lNodeID < aModelData->mNodes.size(); lNodeID++ )
        {
            auto &lNode = aModelData->mNodes[lNodeID];

            if( lNode.mMeshes.size() == 0 ) continue;

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
                    lNodeSkeleton.JointMatrices.push_back( math::mat4( 0.0f ) );
                }

                lNodeSkeleton.BoneCount = lNodeSkeleton.Bones.size();
                for( uint32_t i = 0; i < lNode.mMeshes.size(); i++ )
                {
                    auto &lMesh = lMeshes[lNode.mMeshes[i]];
                    lMesh.Add<sSkeletonComponent>( lNodeSkeleton );
                }
            }
        }

        uint32_t lJointMatrixCount = 0;
        uint32_t lJointOffsetCount = 0;
        ForEach<sSkeletonComponent>(
            [&]( auto lElementToProcess, auto &s )
            {
                lJointMatrixCount += s.JointMatrices.size();
                lJointOffsetCount += 1;
            } );

        mJointTransforms = GPUMemory::Create<math::mat4>( lJointMatrixCount );
        mJointOffsets    = GPUMemory::Create<uint32_t>( lJointOffsetCount );

        if( aModelData->mAnimations.size() > 0 ) lAssetEntity.Add<sAnimationChooser>();

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

        return lAssetEntity;
    }

    void Scene::MarkAsRayTracingTarget( Scene::Element aElement )
    {
        if( !aElement.Has<sStaticMeshComponent>() ) return;

        if( aElement.Has<sRayTracingTargetComponent>() ) return;

        auto &lRTComponent = aElement.Add<sRayTracingTargetComponent>();
    }

    void Scene::AttachScript( Element aElement, std::string aClassName )
    {
        auto &lNewScriptComponent = aElement.Add<sActorComponent>( aClassName );

        lNewScriptComponent.Initialize( aElement );
    }

    void Scene::BeginScenario()
    {
        if( mState != eSceneState::EDITING ) return;

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
                    lComponent.ControllerInstance->OnCreate();
                }
            } );

        ForEach<sActorComponent>( [=]( auto lEntity, auto &lComponent ) { lComponent.OnCreate(); } );

        mState = eSceneState::RUNNING;
    }

    void Scene::EndScenario()
    {
        if( mState != eSceneState::RUNNING ) return;

        ForEach<sAnimatedTransformComponent>(
            [=]( auto lEntity, auto &lComponent )
            { lEntity.AddOrReplace<sNodeTransformComponent>( lEntity.Get<sStaticTransformComponent>().Matrix ); } );

        // Destroy scripts
        ForEach<sBehaviourComponent>(
            [=]( auto lEntity, auto &lComponent )
            {
                if( lComponent.ControllerInstance )
                {
                    lComponent.ControllerInstance->OnDestroy();
                    lComponent.DestroyController( &lComponent );
                }
            } );

        // Destroy Lua scripts
        ForEach<sActorComponent>( [=]( auto lEntity, auto &lComponent ) { lComponent.OnDestroy(); } );

        mState = eSceneState::EDITING;
    }

    void Scene::Update( Timestep ts )
    {
        SE_PROFILE_FUNCTION();

        // Run scripts if the scene is in RUNNING mode.  The native scripts are run first, followed by the Lua scripts.
        if( mState == eSceneState::RUNNING )
        {
            ForEach<sBehaviourComponent>(
                [=]( auto lEntity, auto &lComponent )
                {
                    if( lComponent.ControllerInstance ) lComponent.ControllerInstance->OnUpdate( ts );
                } );

            ForEach<sActorComponent>( [=]( auto lEntity, auto &lComponent ) { lComponent.OnUpdate( ts ); } );

            // Update animations
            ForEach<sAnimationChooser>(
                [=]( auto lEntity, auto &lComponent )
                {
                    auto &lAnimation = lComponent.Animations[0].Get<sAnimationComponent>();
                    lAnimation.CurrentTime += ( ts / 1000.0f );
                    if( lAnimation.CurrentTime > lAnimation.Duration )
                    {
                        lAnimation.CurrentTime -= lAnimation.Duration;
                        lAnimation.CurrentTick = 0;
                    }

                    for( auto &lChannel : lAnimation.mChannels )
                    {
                        auto &lAnimatedTransform = lChannel.mTargetNode.Get<sAnimatedTransformComponent>();
                        if( lAnimation.CurrentTick >= lChannel.mInterpolation.mInputs.size() ) continue;

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

            ForEach<sAnimatedTransformComponent>(
                [&]( auto lElementToProcess, auto &lComponent )
                {
                    math::mat4 lRotation    = math::mat4( lComponent.Rotation );
                    math::mat4 lTranslation = math::Translation( lComponent.Translation );
                    math::mat4 lScale       = math::Scaling( lComponent.Scaling );

                    lElementToProcess.AddOrReplace<sNodeTransformComponent>( lTranslation * lRotation * lScale );
                } );
        }

        std::queue<Entity> lUpdateQueue{};
        lUpdateQueue.push( Root );
        while( !lUpdateQueue.empty() )
        {
            auto lElementToProcess = lUpdateQueue.front();
            lUpdateQueue.pop();

            for( auto lChild : lElementToProcess.Get<sRelationshipComponent>().mChildren ) lUpdateQueue.push( lChild );

            if( lElementToProcess.Has<sNodeTransformComponent>() )
                lElementToProcess.AddOrReplace<sTransformMatrixComponent>( lElementToProcess.Get<sNodeTransformComponent>().mMatrix );

            if( !( lElementToProcess.Get<sRelationshipComponent>().mParent ) ) continue;

            if( !( lElementToProcess.Get<sRelationshipComponent>().mParent.Has<sTransformMatrixComponent>() ) ) continue;

            auto lParent = lElementToProcess.Get<sRelationshipComponent>().mParent;
            if( !( lElementToProcess.Has<sNodeTransformComponent>() ) && !( lElementToProcess.Has<sAnimatedTransformComponent>() ) )
            {
                lElementToProcess.AddOrReplace<sTransformMatrixComponent>( lParent.Get<sTransformMatrixComponent>().Matrix );
            }
            else
            {
                lElementToProcess.AddOrReplace<sTransformMatrixComponent>( lParent.Get<sTransformMatrixComponent>().Matrix *
                                                                           lElementToProcess.Get<sTransformMatrixComponent>().Matrix );
            }
        }

        ForEach<sSkeletonComponent, sTransformMatrixComponent>(
            [&]( auto lElementToProcess, auto &s, auto &t )
            {
                math::mat4 lInverseTransform = math::Inverse( lElementToProcess.Get<sTransformMatrixComponent>().Matrix );

                for( uint32_t lJointID = 0; lJointID < lElementToProcess.Get<sSkeletonComponent>().Bones.size(); lJointID++ )
                {
                    Element    lJoint             = lElementToProcess.Get<sSkeletonComponent>().Bones[lJointID];
                    math::mat4 lInverseBindMatrix = lElementToProcess.Get<sSkeletonComponent>().InverseBindMatrices[lJointID];
                    math::mat4 lJointMatrix       = lJoint.TryGet<sTransformMatrixComponent>( sTransformMatrixComponent{} ).Matrix;
                    lJointMatrix                  = lInverseTransform * lJointMatrix * lInverseBindMatrix;

                    lElementToProcess.Get<sSkeletonComponent>().JointMatrices[lJointID] = lJointMatrix;
                }
            } );

        // if( mVertexBuffer )
        // {
        //     SE_PROFILE_SCOPE( "Transform Vertices" );

        // Update the transformed vertex buffer for static meshies
        {
            std::vector<SE::Cuda::Internal::sGPUDevicePointerView> lVertexBuffers{};
            std::vector<SE::Cuda::Internal::sGPUDevicePointerView> lOutVertexBuffers{};

            std::vector<uint32_t>   lVertexOffsets{};
            std::vector<uint32_t>   lVertexCounts{};
            std::vector<math::mat4> lObjectToWorldTransforms{};
            uint32_t                lMaxVertexCount = 0;
            ForEach<sStaticMeshComponent, sTransformMatrixComponent>(
                [&]( auto aEntiy, auto &aMesh, auto &aTransform )
                {
                    if( aEntiy.Has<sSkeletonComponent>() ) return;

                    lObjectToWorldTransforms.push_back( aTransform.Matrix );
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
                                   mVertexBuffers.DataAs<SE::Cuda::Internal::sGPUDevicePointerView>(),
                                   mTransforms.DataAs<math::mat4>(), lVertexOffsets.size(), mVertexOffsets.DataAs<uint32_t>(),
                                   mVertexCounts.DataAs<uint32_t>(), lMaxVertexCount );
        }

        // Update the transformed vertex buffer for animated meshies
        {
            std::vector<SE::Cuda::Internal::sGPUDevicePointerView> lVertexBuffers{};
            std::vector<SE::Cuda::Internal::sGPUDevicePointerView> lOutVertexBuffers{};

            std::vector<uint32_t>   lVertexOffsets{};
            std::vector<uint32_t>   lVertexCounts{};
            std::vector<math::mat4> lObjectToWorldTransforms{};
            std::vector<math::mat4> lJointTransforms{};
            std::vector<uint32_t>   lJointOffsets{};
            uint32_t                lMaxVertexCount = 0;
            ForEach<sStaticMeshComponent, sTransformMatrixComponent, sSkeletonComponent>(
                [&]( auto aEntiy, auto &aMesh, auto &aTransform, auto &aSkeleton )
                {
                    lObjectToWorldTransforms.push_back( aTransform.Matrix );
                    lVertexBuffers.push_back( *aMesh.mVertexBuffer );
                    lOutVertexBuffers.push_back( *aMesh.mTransformedBuffer );
                    lVertexOffsets.push_back( aMesh.mVertexOffset );
                    lVertexCounts.push_back( aMesh.mVertexCount );
                    lMaxVertexCount = std::max( lMaxVertexCount, static_cast<uint32_t>( aMesh.mVertexCount ) );

                    lJointOffsets.push_back( lJointTransforms.size() );
                    for( auto &lJoint : aSkeleton.JointMatrices ) lJointTransforms.push_back( lJoint );
                } );

            mTransforms.Upload( lObjectToWorldTransforms );
            mVertexBuffers.Upload( lVertexBuffers );
            mTransformedBuffers.Upload( lOutVertexBuffers );
            mVertexOffsets.Upload( lVertexOffsets );
            mVertexCounts.Upload( lVertexCounts );
            mJointOffsets.Upload( lJointOffsets );
            mJointTransforms.Upload( lJointTransforms );

            SkinnedVertexTransform( mTransformedBuffers.DataAs<SE::Cuda::Internal::sGPUDevicePointerView>(),
                                    mVertexBuffers.DataAs<SE::Cuda::Internal::sGPUDevicePointerView>(),
                                    mTransforms.DataAs<math::mat4>(), mJointTransforms.DataAs<math::mat4>(),
                                    mJointOffsets.DataAs<uint32_t>(), lVertexOffsets.size(), mVertexOffsets.DataAs<uint32_t>(),
                                    mVertexCounts.DataAs<uint32_t>(), lMaxVertexCount );
        }

        CUDA_SYNC_CHECK();
        // }

        UpdateRayTracingComponents();
    }

    void Scene::UpdateRayTracingComponents()
    {
        SE_PROFILE_FUNCTION();

        bool lRebuildAS = false;
        ForEach<sTransformMatrixComponent, sStaticMeshComponent, sRayTracingTargetComponent>(
            [&]( auto aEntity, auto &aTransformComponent, auto &aMeshComponent, auto &aRTComponent )
            {
                if( !glm::all( glm::equal( aTransformComponent.Matrix, aRTComponent.Transform, 0.0001 ) ) )
                {
                    aRTComponent.Transform = aTransformComponent.Matrix;
                    lRebuildAS             = true;
                }
            } );

        if( lRebuildAS ) RebuildAccelerationStructure();
    }

    void Scene::RebuildAccelerationStructure()
    {
        SE_PROFILE_FUNCTION();

        mAccelerationStructure = SE::Core::New<OptixScene>( mRayTracingContext );

        ForEach<sRayTracingTargetComponent, sStaticMeshComponent>(
            [&]( auto aEntity, auto &aRTComponent, auto &aMeshComponent )
            {
                mAccelerationStructure->AddGeometry( *aMeshComponent.mTransformedBuffer, *aMeshComponent.mIndexBuffer,
                                                     aMeshComponent.mVertexOffset, aMeshComponent.mVertexCount,
                                                     aMeshComponent.mIndexOffset, aMeshComponent.mIndexCount );
            } );

        mAccelerationStructure->Build();
    }

    void Scene::InitializeRayTracing()
    {
        mRayTracingContext     = SE::Core::New<OptixDeviceContextObject>();
        mAccelerationStructure = SE::Core::New<OptixScene>( mRayTracingContext );
    }

    void Scene::Render() {}

    template <typename ComponentType>
    void WriteComponent( ConfigurationWriter &aOut, Entity const &aEntity )
    {
        if( aEntity.Has<ComponentType>() ) DoWriteComponent( aOut, aEntity.Get<ComponentType>() );
    }

    static void WriteNode( ConfigurationWriter &lOut, Entity const &aEntity, sUUID const &aUUID,
                           std::vector<sImportedAnimationSampler> &lInterpolationData )
    {
        lOut.WriteKey( aUUID.mValue.str() );
        lOut.BeginMap();
        {
            WriteComponent<sTag>( lOut, aEntity );
            WriteComponent<sRelationshipComponent>( lOut, aEntity );
            WriteComponent<sCameraComponent>( lOut, aEntity );
            WriteComponent<sAnimationChooser>( lOut, aEntity );
            WriteComponent<sActorComponent>( lOut, aEntity );

            if( aEntity.Has<sAnimationComponent>() )
            {
                auto &lComponent = aEntity.Get<sAnimationComponent>();
                lOut.WriteKey( "sAnimationComponent" );
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

            WriteComponent<sAnimatedTransformComponent>( lOut, aEntity );
            WriteComponent<sNodeTransformComponent>( lOut, aEntity );
            WriteComponent<sTransformMatrixComponent>( lOut, aEntity );
            WriteComponent<sStaticMeshComponent>( lOut, aEntity );
            WriteComponent<sParticleSystemComponent>( lOut, aEntity );
            WriteComponent<sParticleShaderComponent>( lOut, aEntity );
            WriteComponent<sSkeletonComponent>( lOut, aEntity );
            WriteComponent<sWireframeComponent>( lOut, aEntity );
            WriteComponent<sWireframeMeshComponent>( lOut, aEntity );
            WriteComponent<sBoundingBoxComponent>( lOut, aEntity );
            WriteComponent<sRayTracingTargetComponent>( lOut, aEntity );
            WriteComponent<sMaterialComponent>( lOut, aEntity );
            WriteComponent<sMaterialShaderComponent>( lOut, aEntity );
            WriteComponent<sBackgroundComponent>( lOut, aEntity );
            WriteComponent<sAmbientLightingComponent>( lOut, aEntity );
            WriteComponent<sLightComponent>( lOut, aEntity );
        }
        lOut.EndMap();
    }

    void Scene::SaveAs( fs::path aPath )
    {
        // Check that path does not exist, or exists and is a folder
        if( !fs::exists( aPath ) ) fs::create_directories( aPath );
        if( !fs::is_directory( aPath ) ) return;

        if( !fs::exists( aPath / "Materials" ) ) fs::create_directories( aPath / "Materials" );
        if( !fs::exists( aPath / "Meshes" ) ) fs::create_directories( aPath / "Meshes" );
        if( !fs::exists( aPath / "Animations" ) ) fs::create_directories( aPath / "Animations" );

        {
            auto &lMaterials = mMaterialSystem->GetMaterialData();
            auto &lTextures  = mMaterialSystem->GetTextures();

            for( auto &lMaterial : lMaterials )
            {
                BinaryAsset lBinaryDataFile{};
                std::string lSerializedMeshName = fmt::format( "{}.material", lMaterial.mName );
                auto        lPath               = aPath / "Materials" / lSerializedMeshName;

                auto lRetrieveAndPackageTexture = [&]( sTextureReference aTexture )
                {
                    if( aTextureID < 2 )
                    {
                        sTextureSamplingInfo lSamplingInfo = lTextures[aTexture.mTextureID]->mSpec;
                        TextureData2D        lTextureData;
                        lTextures[aTexture.mTextureID]->GetTexture()->GetPixelData( lTextureData );
                        lBinaryDataFile.Package( lTextureData, lSamplingInfo );
                    }
                };

                sMaterial lNewMaterial          = lMaterial;
                lNewMaterial.mBaseColorTexture  = sTextureReference{ 0, 0 };
                lNewMaterial.mNormalsTexture    = sTextureReference{ 1, 0 };
                lNewMaterial.mMetalRoughTexture = sTextureReference{ 2, 0 };
                lNewMaterial.mOcclusionTexture  = sTextureReference{ 3, 0 };
                lNewMaterial.mEmissiveTexture   = sTextureReference{ 4, 0 };

                lBinaryDataFile.Package( lNewMaterial );

                lRetrieveAndPackageTexture( lMaterial.mBaseColorTexture );
                lRetrieveAndPackageTexture( lMaterial.mNormalsTexture );
                lRetrieveAndPackageTexture( lMaterial.mMetalRoughTexture );
                lRetrieveAndPackageTexture( lMaterial.mOcclusionTexture );
                lRetrieveAndPackageTexture( lMaterial.mEmissiveTexture );

                lBinaryDataFile.WriteTo( lPath );
            }
        }

        ForEach<sStaticMeshComponent>(
            [&]( auto aEntity, auto &aComponent )
            {
                BinaryAsset lBinaryDataFile{};
                std::string lSerializedMeshName = fmt::format( "{}.mesh", aEntity.Get<sUUID>().mValue.str() );

                auto lVertexData = aComponent.mVertexBuffer->Fetch<VertexData>();
                auto lIndexData  = aComponent.mIndexBuffer->Fetch<uint32_t>();
                lBinaryDataFile.Package( lVertexData, lIndexData );
                lBinaryDataFile.WriteTo( aPath / "Meshes" / lSerializedMeshName );
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
            lOut.WriteKey( "nodes" );
            {
                lOut.BeginMap();
                ForEach<sUUID>( [&]( auto aEntity, auto &aUUID ) { WriteNode( lOut, aEntity, aUUID, lInterpolationData ); } );
                lOut.EndMap();
            }
            lOut.EndMap();
        }
        lOut.EndMap();

        fs::path    lOutput = aPath / "Animations" / "BinaryData.bin";
        BinaryAsset lBinaryDataFile;
        for( auto &lInterpolation : lInterpolationData ) lBinaryDataFile.Package( lInterpolation );
        lBinaryDataFile.WriteTo( lOutput );
    }
} // namespace SE::Core
