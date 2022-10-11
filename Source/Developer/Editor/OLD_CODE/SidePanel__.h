#pragma once
#include <string>

namespace LTSE::Editor
{

    class SidePanel
    {
      public:
        std::string Title = "";

      public:
        SidePanel()  = default;
        ~SidePanel() = default;

        void Display( int32_t width, int32_t height );
    };

} // namespace LTSE::Editor