
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

#include "Renderer/MaterialSystem.h"

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

        Scene( ref_t<IGraphicContext> a_GraphicContext, ref_t<SE::Core::UIContext> a_UI );
        Scene( ref_t<Scene> aSource );
        Scene( Scene & ) = delete;
        ~Scene();

        Element Create( string_t a_Name, Element a_Parent );

        Element CreateEntity();
        Element CreateEntity( string_t a_Name );

        void LoadScenario( fs::path aScenarioPath );

        Element LoadModel( ref_t<sImportedModel> aModelData, mat4 aTransform );
        Element LoadModel( ref_t<sImportedModel> aModelData, mat4 aTransform, string_t a_Name );

        void SaveAs( fs::path aPath );

        void BeginScenario();
        void EndScenario();

        void MarkAsRayTracingTarget( Element a_Element );
        void AttachScript( Element aElement, string_t aScriptPath );

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

        ref_t<IGraphicContext> GetGraphicContext()
        {
            return mGraphicContext;
        }

        template <typename... Args>
        void ForEach( std::function<void( Element, Args &... )> a_ApplyFunction )
        {
            mRegistry.ForEach<Args...>( a_ApplyFunction );
        }

        void UpdateRayTracingComponents();

        OptixTraversableHandle GetRayTracingRoot()
        {
            if( mAccelerationStructure )
                return mAccelerationStructure->mOptixObject;
            return 0;
        }

        ref_t<SE::Graphics::OptixDeviceContextObject> GetRayTracingContext()
        {
            return mRayTracingContext;
        }

        eSceneState GetState()
        {
            return mState;
        }

        // ref_t<MaterialSystem> GetMaterialSystem()
        // {
        //     return mMaterialSystem;
        // }

        ref_t<MaterialSystem> GetMaterialSystem()
        {
            return mMaterialSystem;
        }

        void ClearScene();

        void SetViewport( vec2 aPosition, vec2 aSize );

        mat4 mEditorView;

      private:
        eSceneState                   mState                 = eSceneState::EDITING;
        ref_t<IGraphicContext>          mGraphicContext        = nullptr;
        ref_t<OptixDeviceContextObject> mRayTracingContext     = nullptr;
        ref_t<OptixScene>               mAccelerationStructure = nullptr;

        // Handle to the new version of the material system
        ref_t<MaterialSystem> mMaterialSystem = nullptr;

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
