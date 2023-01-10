
#pragma once

#include <filesystem>
#include <optional>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "UI/UI.h"

#include "Core/GraphicContext//UI/UIContext.h"
#include "VertexData.h"

#include "Core/CUDA/Texture/TextureTypes.h"
#include "Core/EntityRegistry/Registry.h"

#include "Core/Optix/OptixAccelerationStructure.h"
#include "Core/Optix/OptixContext.h"
#include "Core/Optix/OptixModule.h"
#include "Core/Optix/OptixPipeline.h"
#include "Core/Optix/OptixShaderBindingTable.h"

#include "Importer/ImporterData.h"

#include "Components.h"

using namespace math;
using namespace math::literals;
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

        Scene( Ref<VkGraphicContext> a_GraphicContext, Ref<SE::Core::UIContext> a_UI );
        Scene( Ref<Scene> aSource );
        Scene( Scene & ) = delete;
        ~Scene();

        Element Create( std::string a_Name, Element a_Parent );

        Element CreateEntity();
        Element CreateEntity( std::string a_Name );

        void LoadScenario( fs::path aScenarioPath );

        Element LoadModel( Ref<sImportedModel> aModelData, math::mat4 aTransform );
        Element LoadModel( Ref<sImportedModel> aModelData, math::mat4 aTransform, std::string a_Name );

        void SaveAs( fs::path aPath );

        void BeginScenario();
        void EndScenario();

        void MarkAsRayTracingTarget( Element a_Element );
        void AttachScript( Element aElement, std::string aScriptPath );

        math::mat4 GetView();
        math::mat4 GetProjection();
        math::vec3 GetCameraPosition();

        void Update( Timestep ts );
        void Render();

        Element CurrentCamera;
        Element DefaultCamera;

        Element Environment;
        Element Root;

        Ref<VkGraphicContext> GetGraphicContext() { return mGraphicContext; }

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

        Ref<VkGpuBuffer> mVertexBuffer            = nullptr;
        Ref<VkGpuBuffer> mIndexBuffer             = nullptr;
        Ref<VkGpuBuffer> mTransformedVertexBuffer = nullptr;

      private:
        eSceneState           mState = eSceneState::EDITING;
        Ref<VkGraphicContext> mGraphicContext;
        Ref<MaterialSystem>   mMaterialSystem;

        Ref<OptixDeviceContextObject> mRayTracingContext     = nullptr;
        Ref<OptixScene>               mAccelerationStructure = nullptr;

        std::vector<sActorComponent> mActorComponents;

      protected:
        SE::Core::EntityRegistry mRegistry;

        void InitializeRayTracing();
        void RebuildAccelerationStructure();
        void DestroyEntity( Element entity );
        void ConnectSignalHandlers();

        GPUMemory mTransforms{};
        GPUMemory mVertexBuffers{};
        GPUMemory mTransformedBuffers{};
        GPUMemory mVertexOffsets{};
        GPUMemory mVertexCounts{};
        GPUMemory mJointTransforms{};
        GPUMemory mJointOffsets{};

        bool mIsClone = false;

      private:
        friend class Element;
    };

} // namespace SE::Core
