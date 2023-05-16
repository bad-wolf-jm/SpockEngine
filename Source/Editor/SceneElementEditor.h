#pragma once
#include <string>

#include "Core/Entity/Collection.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Scene/Scene.h"

#include "Engine/Engine.h"

using namespace SE::Core;
using namespace SE::Graphics;
using namespace SE::Core::EntityComponentSystem;

namespace SE::Editor
{
    class SceneElementEditor
    {
      public:
        Entity     ElementToEdit{};
        Ref<Scene> World;

      public:
        SceneElementEditor() = default;
        SceneElementEditor( SE::Graphics::Ref<VkGraphicContext> aGraphicContext );
        ~SceneElementEditor() = default;

        void Display( int32_t width, int32_t height );

      private:
        SE::Graphics::Ref<VkGraphicContext> mGraphicContext;
    };

} // namespace SE::Editor