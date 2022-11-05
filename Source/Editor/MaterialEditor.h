#pragma once
#include <string>

#include "Core/Memory.h"
#include "Core/EntityRegistry/Registry.h"

#include "Core/GraphicContext//GraphicContext.h"
#include "Scene/Scene.h"

#include "Engine/Engine.h"

using namespace LTSE::Core;

namespace LTSE::Editor
{
    class MaterialEditor
    {
      public:
        Entity ElementToEdit{};
        Ref<Scene> World;

      public:
        MaterialEditor()  = default;
        ~MaterialEditor() = default;

        void Display( int32_t width, int32_t height );
    };

} // namespace LTSE::Editor