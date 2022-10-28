
#pragma once

#include <filesystem>
#include <optional>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/Cuda/ExternalMemory.h"

#include "Graphics/API/DescriptorSet.h"
#include "Graphics/API/GraphicContext.h"
#include "Graphics/API/GraphicsPipeline.h"

#include "UI/UI.h"

#include "Graphics/API/UI/UIContext.h"
#include "VertexData.h"

#include "Core/EntityRegistry/Registry.h"
#include "Core/Textures/TextureTypes.h"

#include "Scripting/ScriptingEngine.h"

#include "Core/Optix/OptixAccelerationStructure.h"
#include "Core/Optix/OptixContext.h"
#include "Core/Optix/OptixModule.h"
#include "Core/Optix/OptixPipeline.h"
#include "Core/Optix/OptixShaderBindingTable.h"

#include "Importer/ImporterData.h"

#include "Components.h"

using namespace math;
using namespace math::literals;
using namespace LTSE::Graphics;
namespace fs = std::filesystem;
using namespace LTSE::Core::EntityComponentSystem::Components;

namespace LTSE::Core
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

        Scene( GraphicContext &a_GraphicContext, Ref<LTSE::Core::UIContext> a_UI );
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
        void AttachScript( Element aElement, fs::path aScriptPath );

        math::mat4 GetView();
        math::mat4 GetProjection();
        math::vec3 GetCameraPosition();

        void Update( Timestep ts );
        void Render();

        Element CurrentCamera;
        Element DefaultCamera;

        Element Environment;
        Element Root;

        GraphicContext &GetGraphicContext() { return mGraphicContext; }

        template <typename... Args>
        void ForEach( std::function<void( Element, Args &... )> a_ApplyFunction )
        {
            m_Registry.ForEach<Args...>( a_ApplyFunction );
        }

        void                   UpdateRayTracingComponents();
        OptixTraversableHandle GetRayTracingRoot()
        {
            if( m_AccelerationStructure ) return m_AccelerationStructure->RTObject;
            return 0;
        }

        Ref<LTSE::Graphics::OptixDeviceContextObject> GetRayTracingContext() { return mRayTracingContext; }

        eSceneState         GetState() { return mState; }
        Ref<MaterialSystem> GetMaterialSystem() { return mMaterialSystem; }

        void ClearScene();

        Ref<Buffer> mVertexBuffer = nullptr;
        Ref<Buffer> mIndexBuffer  = nullptr;

        Ref<Buffer> mTransformedVertexBuffer = nullptr;

        Cuda::GPUExternalMemory mTransformedVertexBufferMemoryHandle{};
        Cuda::GPUExternalMemory mVertexBufferMemoryHandle{};
        Cuda::GPUExternalMemory mIndexBufferMemoryHandle{};

      private:
        eSceneState         mState = eSceneState::EDITING;
        GraphicContext      mGraphicContext;
        Ref<MaterialSystem> mMaterialSystem;

        Ref<ScriptingEngine> mSceneScripting = nullptr;

        Ref<UIContext>                m_UI                    = nullptr;
        Ref<OptixDeviceContextObject> mRayTracingContext     = nullptr;
        Ref<OptixTraversableObject>   m_AccelerationStructure = nullptr;

        Ref<Graphics::Texture2D> m_EmptyTexturePreview = nullptr;
        UI::ImageHandle          m_EmptyTexturePreviewImageHandle;

        void UpdateParent( Entity const &aEntity, sRelationshipComponent const &aComponent );
        void UpdateLocalTransform( Entity const &aEntity, sNodeTransformComponent const &aComponent );
        void UpdateTransformMatrix( Entity const &aEntity, sTransformMatrixComponent const &aComponent );

      protected:
        LTSE::Core::EntityRegistry m_Registry;

        void InitializeRayTracing();
        void RebuildAccelerationStructure();
        void DestroyEntity( Element entity );
        void ConnectSignalHandlers();

        // std::unordered_map<std::string, Element> mAssetSystem = {};
        GPUMemory mTransforms{};
        GPUMemory mVertexOffsets{};
        GPUMemory mVertexCounts{};
        GPUMemory mJointTransforms{};
        GPUMemory mJointOffsets{};

        bool mIsClone = false;

      private:
        friend class Element;
    };

} // namespace LTSE::Core
