#pragma once
#include <string>

// #include "SidePanel.h"

#include "Core/Entity/Collection.h"
#include "Core/Memory.h"

#include "Scene/Scene.h"

#include "SceneElementEditor.h"

namespace SE::Editor
{
    using namespace SE::Core;

    struct LockComponent
    {
        bool mLocked = true;
    };

    class SceneHierarchyPanel
    {
      public:
        Ref<Scene> World = nullptr;
        Entity SelectedElement{};
        SceneElementEditor ElementEditor{};
        bool RequestEditSceneElement = false;

      public:
        SceneHierarchyPanel()  = default;
        ~SceneHierarchyPanel() = default;

        void Display( int32_t width, int32_t height );

      private:
        void DisplayNode( Scene::Element a_Node, float a_Width );
    };

} // namespace SE::Editor