#pragma once
#include <string>

#include "Core/EntityRegistry/Registry.h"
#include "Core/Memory.h"

#include "Developer/GraphicContext/GraphicContext.h"
#include "Developer/Scene/Scene.h"

#include "Developer/Platform/EngineLoop.h"

using namespace LTSE::Core;
using namespace LTSE::Graphics;
using namespace LTSE::Core::EntityComponentSystem;

namespace LTSE::Editor
{
    class SceneElementEditor
    {
      public:
        Entity ElementToEdit{};
        Ref<Scene> World;

      public:
        SceneElementEditor() = default;
        SceneElementEditor( LTSE::Graphics::GraphicContext &aGraphicContext );
        ~SceneElementEditor() = default;

        void Display( int32_t width, int32_t height );

      private:
        LTSE::Graphics::GraphicContext mGraphicContext;
    };

} // namespace LTSE::Editor