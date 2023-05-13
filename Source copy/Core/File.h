#pragma once

#include <GLFW/glfw3.h>
#include <optional>
#include <string>


class FileDialogs
{
  public:
    // These return empty strings if cancelled
    static std::optional<std::string> OpenFile( GLFWwindow *owner, const char *filter );
    static std::optional<std::string> SaveFile( GLFWwindow *owner, const char *filter );
};
