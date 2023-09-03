#include "File.h"

#ifdef APIENTRY
#    undef APIENTRY
#endif

// #include <GLFW/glfw3.h>
#include <commdlg.h>
#include <sstream>
#include <windows.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

namespace SE::Core
{
    std::optional<string_t> FileDialogs::OpenFile( GLFWwindow *owner, const char *filter )
    {
        OPENFILENAMEA ofn;
        CHAR          szFile[260] = { 0 };
        ZeroMemory( &ofn, sizeof( OPENFILENAME ) );
        ofn.lStructSize  = sizeof( OPENFILENAME );
        ofn.hwndOwner    = glfwGetWin32Window( owner );
        ofn.lpstrFile    = szFile;
        ofn.nMaxFile     = sizeof( szFile );
        ofn.lpstrFilter  = filter;
        ofn.nFilterIndex = 1;
        ofn.Flags        = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

        if( GetOpenFileNameA( &ofn ) == TRUE )
            return ofn.lpstrFile;
        return std::nullopt;
    }

    std::optional<string_t> FileDialogs::SaveFile( GLFWwindow *owner, const char *filter )
    {
        OPENFILENAMEA ofn;
        CHAR          szFile[260] = { 0 };
        ZeroMemory( &ofn, sizeof( OPENFILENAME ) );
        ofn.lStructSize  = sizeof( OPENFILENAME );
        ofn.hwndOwner    = glfwGetWin32Window( owner );
        ofn.lpstrFile    = szFile;
        ofn.nMaxFile     = sizeof( szFile );
        ofn.lpstrFilter  = filter;
        ofn.nFilterIndex = 1;
        ofn.Flags        = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

        // Sets the default extension by extracting it from the filter
        ofn.lpstrDefExt = strchr( filter, '\0' ) + 1;

        if( GetSaveFileNameA( &ofn ) == TRUE )
            return ofn.lpstrFile;

        return std::nullopt;
    }
} // namespace SE::Core
