#pragma once
#include <string>

namespace LTSE::Editor
{

    class PropertiesPanel
    {
      public:
        std::string Title = "";

      public:
        PropertiesPanel()  = default;
        ~PropertiesPanel() = default;

        void Display( int32_t width, int32_t height );
    };

} // namespace LTSE::Editor