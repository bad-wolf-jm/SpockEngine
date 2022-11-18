#pragma once
#include <string>

#include "Core/EntityRegistry/Registry.h"
#include "Core/Memory.h"

#include "Core/GraphicContext//GraphicContext.h"
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
        Entity ElementToEdit{};
        Ref<Scene> World;

      public:
        SceneElementEditor() = default;
        SceneElementEditor( SE::Graphics::GraphicContext &aGraphicContext );
        ~SceneElementEditor() = default;

        void Display( int32_t width, int32_t height );

      private:
        SE::Graphics::GraphicContext mGraphicContext;
    };

} // namespace SE::Editor