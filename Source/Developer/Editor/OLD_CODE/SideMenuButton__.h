#pragma once
#include <string>

namespace LTSE::Editor
{

    class SideMenuButton
    {
      public:
        std::string Title = "";

        bool Clicked  = false;
        bool Selected = false;
        bool Hovered  = false;

      public:
        SideMenuButton()  = default;
        ~SideMenuButton() = default;

        void Display( int32_t width, int32_t height );
    };

} // namespace LTSE::Editor