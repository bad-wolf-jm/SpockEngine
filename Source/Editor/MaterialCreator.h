#pragma once

#include "Core/EntityRegistry/Registry.h"


#include "PopupWindow.h"

using namespace LTSE::Core;

namespace LTSE::Editor
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

} // namespace LTSE::Editor