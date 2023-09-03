#pragma once
#include <string>

#include "Core/Entity/Collection.h"
#include "Core/Memory.h"

#include "Graphics/Api.h"
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
        ref_t<Scene> World;

      public:
        SceneElementEditor() = default;
        SceneElementEditor( ref_t<IGraphicContext> aGraphicContext );
        ~SceneElementEditor() = default;

        void Display( int32_t width, int32_t height );

      private:
        ref_t<IGraphicContext> mGraphicContext;
    };

} // namespace SE::Editor