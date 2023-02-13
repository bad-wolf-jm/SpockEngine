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

#include "OtdrScene.h"

#include "Scene/Serialize/SerializeComponents.h"

namespace SE::Core
{

    namespace fs = std::filesystem;
    using namespace SE::Graphics;
    using namespace SE::Cuda;

    OtdrScene::OtdrScene()
    {
        Root = mRegistry.CreateEntityWithRelationship( "WorldRoot" );

        ConnectSignalHandlers();
    }

    template <typename _Component>
    static void CopyComponent( Entity &aSource, Entity &aDestination )
    {
        if( ( aSource.Has<_Component>() ) ) aDestination.AddOrReplace<_Component>( aSource.Get<_Component>() );
    }

    OtdrScene::OtdrScene( Ref<OtdrScene> aSource )
    {
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

        mIsClone = true;
    }

    OtdrScene::~OtdrScene() {}

    void OtdrScene::SetViewport( math::vec2 aPosition, math::vec2 aSize )
    {
        mViewportPosition = aPosition;
        mViewportSize     = aSize;
    }

    OtdrScene::Element OtdrScene::Create( std::string aName, Element aParent ) { return mRegistry.CreateEntity( aParent, aName ); }

    OtdrScene::Element OtdrScene::CreateEntity() { return mRegistry.CreateEntity(); }

    OtdrScene::Element OtdrScene::CreateEntity( std::string aName ) { return mRegistry.CreateEntity( aName ); }

    void OtdrScene::ClearScene()
    {
        mRegistry.Clear();

        Root = mRegistry.CreateEntityWithRelationship( "WorldRoot" );
    }

    void OtdrScene::ConnectSignalHandlers()
    {
        // clang-format off
        mRegistry.OnComponentAdded<sActorComponent>( [&]( auto aEntity, auto &aComponent ) { 
            aComponent.Initialize( aEntity ); 
        } );

        mRegistry.OnComponentUpdated<sActorComponent>( [&]( auto aEntity, auto &aComponent ) { 
            aComponent.Initialize( aEntity ); 
        } );

        mRegistry.OnComponentAdded<sHUDComponent>( [&]( auto aEntity, auto &aComponent ) { 
            aComponent.Initialize( aEntity ); 
        } );

        mRegistry.OnComponentUpdated<sHUDComponent>( [&]( auto aEntity, auto &aComponent ) { 
            aComponent.Initialize( aEntity ); 
        } );
        // clang-format on
    }

    void OtdrScene::LoadScenario( fs::path aScenarioPath )
    {
        mRegistry.Clear();

        ConnectSignalHandlers();

        auto lScenarioRoot = aScenarioPath.parent_path();

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

            if( HasTypeTag<sTag>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sTag>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sTag>()], lReadContext );
            }

            if( HasTypeTag<sActorComponent>( lEntityConfiguration ) )
            {
                auto &lComponent = lEntity.Add<sActorComponent>();

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sActorComponent>()], lReadContext );
            }

            if( HasTypeTag<sHUDComponent>( lEntityConfiguration ) )
            {
                sHUDComponent lComponent{};

                ReadComponent( lComponent, lEntityConfiguration[TypeTag<sHUDComponent>()], lReadContext );
                lEntity.Add<sHUDComponent>( lComponent );
            }
        }

        for( YAML::iterator it = lNodesRoot.begin(); it != lNodesRoot.end(); ++it )
        {
            auto const &lKey                 = it->first.as<std::string>();
            auto       &lEntityConfiguration = it->second;

            auto &lEntity = lReadContext.mEntities[lKey];

            if( !lEntity ) return;

            if( lParentEntityLUT.find( lKey ) != lParentEntityLUT.end() )
                mRegistry.SetParent( lEntity, lReadContext.mEntities[lParentEntityLUT[lKey]] );
        }

        auto lRootNodeUUIDStr = Get( lSceneRoot["root"], std::string{ "" } );
        auto lRootNodeUUID    = UUIDv4::UUID::fromStrFactory( lRootNodeUUIDStr );
        Root                  = lReadContext.mEntities[lRootNodeUUIDStr];
        SE::Logging::Info( "Created root", lRootNodeUUIDStr );
    }

    void OtdrScene::AttachScript( Element aElement, std::string aClassName )
    {
        auto &lNewScriptComponent = aElement.Add<sActorComponent>( aClassName );
    }

    void OtdrScene::BeginScenario()
    {
        if( mState != eSceneState::EDITING ) return;

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
        ForEach<sHUDComponent>( [=]( auto lEntity, auto &lComponent ) { lComponent.OnCreate(); } );

        mState = eSceneState::RUNNING;
    }

    void OtdrScene::EndScenario()
    {
        if( mState != eSceneState::RUNNING ) return;

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

    void OtdrScene::Update( Timestep ts )
    {
        SE_PROFILE_FUNCTION();

        // Run scripts if the scene is in RUNNING mode.  The native scripts are run first, followed by the Lua scripts.
        if( mState == eSceneState::RUNNING )
        {
            ForEach<sBehaviourComponent>(
                [=]( auto aEntity, auto &aComponent )
                {
                    if( aComponent.ControllerInstance ) aComponent.ControllerInstance->OnUpdate( ts );
                } );

            ForEach<sActorComponent>( [=]( auto lEntity, auto &lComponent ) { lComponent.OnUpdate( ts ); } );
        }

        ForEach<sHUDComponent>(
            [=]( auto aEntity, auto &aComponent )
            {
                if( ( mState != eSceneState::RUNNING ) && !aComponent.mDisplayInEditor ) return;

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
    }

    static void WriteNode( ConfigurationWriter &lOut, Entity const &aEntity, sUUID const &aUUID )
    {
        lOut.WriteKey( aUUID.mValue.str() );
        lOut.BeginMap();
        {
            if( aEntity.Has<sTag>() )
            {
                WriteComponent( lOut, aEntity.Get<sTag>() );
            }
            if( aEntity.Has<sRelationshipComponent>() ) WriteComponent( lOut, aEntity.Get<sRelationshipComponent>() );
            if( aEntity.Has<sActorComponent>() ) WriteComponent( lOut, aEntity.Get<sActorComponent>() );
            if( aEntity.Has<sHUDComponent>() ) WriteComponent( lOut, aEntity.Get<sHUDComponent>() );
        }
        lOut.EndMap();
    }

    void OtdrScene::SaveAs( fs::path aPath )
    {
        // Check that path does not exist, or exists and is a folder
        if( !fs::exists( aPath ) ) fs::create_directories( aPath );
        if( !fs::is_directory( aPath ) ) return;

        auto lOut = ConfigurationWriter( aPath / "SceneDefinition.yaml" );

        std::vector<sImportedAnimationSampler> lInterpolationData;
        lOut.BeginMap();
        lOut.WriteKey( "scene" );
        {
            lOut.BeginMap();
            lOut.WriteKey( "name", "FOO" );
            lOut.WriteKey( "version", "1" );
            lOut.WriteKey( "root", Root.Get<sUUID>().mValue.str() );
            lOut.WriteKey( "nodes" );
            {
                lOut.BeginMap();
                ForEach<sUUID>( [&]( auto aEntity, auto &aUUID ) { WriteNode( lOut, aEntity, aUUID ); } );
                lOut.EndMap();
            }
            lOut.EndMap();
        }
        lOut.EndMap();
    }
} // namespace SE::Core
