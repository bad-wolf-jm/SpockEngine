
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

#include "Core/EntityCollection/Collection.h"

using namespace math;
using namespace math::literals;
using namespace SE::Graphics;
namespace fs = std::filesystem;

namespace SE::Core
{
    class OtdrScene
    {
      public:
        enum class eSceneState : uint8_t
        {
            EDITING,
            RUNNING
        };

        typedef Entity Element;

        OtdrScene();
        OtdrScene( Ref<OtdrScene> aSource );
        OtdrScene( OtdrScene & ) = delete;
        ~OtdrScene();

        Element Create( std::string a_Name, Element a_Parent );
        Element CreateEntity();
        Element CreateEntity( std::string a_Name );

        void Load( fs::path aScenarioPath );
        void SaveAs( fs::path aPath );

        void BeginScenario();
        void EndScenario();
        void AttachScript( Element aElement, std::string aScriptPath );
        void Update( Timestep ts );

        Element Root;

        template <typename... Args>
        void ForEach( std::function<void( Element, Args &... )> a_ApplyFunction )
        {
            mRegistry.ForEach<Args...>( a_ApplyFunction );
        }

        eSceneState GetState() { return mState; }

        void ClearScene();

        void SetViewport( math::vec2 aPosition, math::vec2 aSize );

      private:
        eSceneState mState = eSceneState::EDITING;

      protected:
        SE::Core::EntityCollection mRegistry;

        void DestroyEntity( Element entity );
        void ConnectSignalHandlers();

        bool mIsClone = false;

        math::vec2 mViewportPosition{};
        math::vec2 mViewportSize{};

      private:
        friend class Element;
    };

} // namespace SE::Core
