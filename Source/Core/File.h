#pragma once

#include <GLFW/glfw3.h>
#include <optional>
#include <string>

#include "Core/String.h"

using namespace SE::Core;

class FileDialogs
{
  public:
    // These return empty strings if cancelled
    static std::optional<string_t> OpenFile( GLFWwindow *owner, const char *filter );
    static std::optional<std::wstring> OpenFile( GLFWwindow *owner, const wchar_t *filter );
    static std::optional<string_t> SaveFile( GLFWwindow *owner, const char *filter );
};
