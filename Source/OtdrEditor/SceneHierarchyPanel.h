#pragma once
#include <string>

// #include "SidePanel.h"

#include "Core/EntityCollection/Collection.h"
#include "Core/Memory.h"

#include "OtdrScene/OtdrScene.h"

#include "SceneElementEditor.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    struct LockComponent
    {
        bool mLocked = true;
    };

    class OtdrSceneHierarchyPanel
    {
      public:
        Ref<OtdrScene> World = nullptr;
        Entity SelectedElement{};
        // SceneElementEditor ElementEditor{};
        bool RequestEditSceneElement = false;

      public:
        OtdrSceneHierarchyPanel()  = default;
        ~OtdrSceneHierarchyPanel() = default;

        void Display( int32_t width, int32_t height );

      private:
        void DisplayNode( OtdrScene::Element a_Node, float a_Width );
    };

} // namespace SE::Editor