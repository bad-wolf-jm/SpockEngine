#pragma once
#include <string>

#include "Core/EntityRegistry/Registry.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Scene/Scene.h"

#include "Engine/Engine.h"

using namespace SE::Core;

namespace SE::Editor
{
    class MaterialEditor
    {
      public:
        Entity     ElementToEdit{};
        Ref<Scene> World;

      public:
        MaterialEditor()  = default;
        ~MaterialEditor() = default;

        void Display( int32_t width, int32_t height );
    };

} // namespace SE::Editor