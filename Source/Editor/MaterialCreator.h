#pragma once

#include "Core/EntityCollection/Collection.h"

#include "PopupWindow.h"

using namespace SE::Core;

namespace SE::Editor
{
    class MaterialCreator : public PopupWindow
    {
      public:
        Entity NewMaterial{};

      public:
        MaterialCreator() = default;
        MaterialCreator( std::string a_Title, math::vec2 a_Size );
        ~MaterialCreator() = default;

        void WindowContent();
    };

} // namespace SE::Editor