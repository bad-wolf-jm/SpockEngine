
#pragma once

#include <filesystem>
#include <optional>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "UI/UI.h"

#include "UI/UIContext.h"
#include "VertexData.h"

#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/Entity/Collection.h"

#include "Core/Optix/OptixAccelerationStructure.h"
#include "Core/Optix/OptixContext.h"
#include "Core/Optix/OptixModule.h"
#include "Core/Optix/OptixPipeline.h"
#include "Core/Optix/OptixShaderBindingTable.h"

#include "Importer/ImporterData.h"

#include "Components.h"

// #include "Renderer2/SceneRenderer.h"

using namespace math;
using namespace literals;
using namespace SE::Graphics;
namespace fs = std::filesystem;
using namespace SE::Core::EntityComponentSystem::Components;

namespace SE::Core
{
    class Scene
    {
      public:
        enum class eSceneState : uint8_t
        {
            EDITING,
            RUNNING
        };

        typedef Entity Element;

        Scene( Ref<IGraphicContext> a_GraphicContext, Ref<SE::Core::UIContext> a_UI );
        Scene( Ref<Scene> aSource );
        Scene( Scene & ) = delete;
        ~Scene();

        Element Create( std::string a_Name, Element a_Parent );

        Element CreateEntity();
        Element CreateEntity( std::string a_Name );

        void LoadScenario( fs::path aScenarioPath );

        Element LoadModel( Ref<sImportedModel> aModelData, mat4 aTransform );
        Element LoadModel( Ref<sImportedModel> aModelData, mat4 aTransform, std::string a_Name );

        void SaveAs( fs::path aPath );

        void BeginScenario();
        void EndScenario();

        void MarkAsRayTracingTarget( Element a_Element );
        void AttachScript( Element aElement, std::string aScriptPath );

        mat4 GetView();
        mat4 GetProjection();
        vec3 GetCameraPosition();
        mat4 GetFinalTransformMatrix( Element aElement );

        void Update( Timestep ts );
        void UpdateAnimation( Entity &aAnimation, Timestep const &ts );
        void Render();

        Element Root;
        Element Environment;
        Element CurrentCamera;
        Element DefaultCamera;

        Ref<IGraphicContext> GetGraphicContext() { return mGraphicContext; }

        template <typename... Args>
        void ForEach( std::function<void( Element, Args &... )> a_ApplyFunction )
        {
            mRegistry.ForEach<Args...>( a_ApplyFunction );
        }

        void UpdateRayTracingComponents();

        OptixTraversableHandle GetRayTracingRoot()
        {
            if( mAccelerationStructure ) return mAccelerationStructure->mOptixObject;
            return 0;
        }

        Ref<SE::Graphics::OptixDeviceContextObject> GetRayTracingContext() { return mRayTracingContext; }

        eSceneState         GetState() { return mState; }
        Ref<MaterialSystem> GetMaterialSystem() { return mMaterialSystem; }

        void ClearScene();

        void SetViewport( vec2 aPosition, vec2 aSize );

        mat4 mEditorView;

      private:
        eSceneState                   mState                 = eSceneState::EDITING;
        Ref<IGraphicContext>          mGraphicContext        = nullptr;
        Ref<MaterialSystem>           mMaterialSystem        = nullptr;
        Ref<OptixDeviceContextObject> mRayTracingContext     = nullptr;
        Ref<OptixScene>               mAccelerationStructure = nullptr;

        // Handle to the new version of the renderer
        // Ref<SceneRenderer> mRenderer = nullptr;

        std::vector<sActorComponent> mActorComponents;

      protected:
        SE::Core::EntityCollection mRegistry;

        void InitializeRayTracing();
        void RebuildAccelerationStructure();
        void DestroyEntity( Element entity );
        void ConnectSignalHandlers();

        void ResizeCUDABuffers();

        GPUMemory mTransforms{};
        GPUMemory mVertexBuffers{};
        GPUMemory mTransformedBuffers{};
        GPUMemory mVertexOffsets{};
        GPUMemory mVertexCounts{};
        GPUMemory mJointTransforms{};
        GPUMemory mJointOffsets{};

        bool mIsClone = false;

        vec2 mViewportPosition{};
        vec2 mViewportSize{};

      private:
        friend class Element;

        std::unordered_map<UUIDv4::UUID, mat4> mTransformCache{};
    };

} // namespace SE::Core
