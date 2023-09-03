#pragma once

#include "Core/String.h"
#include <GLFW/glfw3.h>
#include <optional>
#include <string>

namespace SE::Core
{
    class FileDialogs
    {
      public:
        // These return empty strings if cancelled
        static std::optional<string_t> OpenFile( GLFWwindow *owner, const char *filter );
        static std::optional<string_t> SaveFile( GLFWwindow *owner, const char *filter );
    };

} // namespace SE::Core
