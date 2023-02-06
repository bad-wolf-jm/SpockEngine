#include "File.h"

#ifdef APIENTRY
#    undef APIENTRY
#endif

// clang-format off
#include <sstream>
#include <windows.h>
#include <commdlg.h>
// clang-format on 

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <fstream>

namespace SE::Core
{
    std::optional<std::string> FileDialogs::OpenFile( GLFWwindow *owner, const char *filter )
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

        if( GetOpenFileNameA( &ofn ) == TRUE ) return ofn.lpstrFile;
        return std::nullopt;
    }

    std::optional<std::string> FileDialogs::SaveFile( GLFWwindow *owner, const char *filter )
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

        if( GetSaveFileNameA( &ofn ) == TRUE ) return ofn.lpstrFile;
        return std::nullopt;
    }

    std::vector<char> &ReadFile( const std::string &aFilename )
    {
        std::ifstream lFileObject( aFilename, std::ios::ate | std::ios::binary );

        if( !lFileObject.is_open() ) throw std::runtime_error( "Failed to open file!" );

        size_t lFileSize = (size_t)lFileObject.tellg();

        std::vector<char> lBuffer( lFileSize );
        lFileObject.seekg( 0 );
        lFileObject.read( lBuffer.data(), lFileSize );
        lFileObject.close();

        return std::move(lBuffer);
    }
} // namespace SE::Core
