#pragma once

#include <fmt/core.h>

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"
#include "implot.h"

#include "Graphics/API/DescriptorSet.h"
#include "Graphics/API/UI/FontAwesome.h"
#include "Core/Memory.h"
#include "Core/Types.h"


namespace LTSE::Core::UI
{
    class UIStyle
    {
      public:
        UIStyle() = default;
        UIStyle( [[maybe_unused]] bool x );
        ~UIStyle() = default;

      private:
    };

    struct ImageHandle
    {
        Ref<LTSE::Graphics::DescriptorSet> Handle = nullptr;

        ImageHandle()                      = default;
        ImageHandle( const ImageHandle & ) = default;
    };

    void Text( const char *a_Text );
    void Text( std::string a_Text );

    template <typename... ArgTypes>
    void Text( std::string a_String, ArgTypes &&...a_ArgList )
    {
        Text( fmt::format( a_String, std::forward<ArgTypes>( a_ArgList )... ) );
    }

    void SameLine();
    void SameLine( float spacing );

    math::ivec2 GetRootWindowSize();
    math::vec2  GetCurrentCursorPosition();
    math::vec2  GetCurrentCursorScreenPosition();
    math::ivec2 GetAvailableContentSpace();
    math::ivec2 GetCurrentWindowPosition();

    void SetCursorPosition( math::vec2 a_Position );
    void SetCursorPosition( ImVec2 a_Position );
    void SetCursorPositionX( float a_Position );
    void SetCursorPositionY( float a_Position );

    void SetNextWindowPosition( math::vec2 a_Position );
    void SetNextWindowSize( math::vec2 a_Size );

} // namespace LTSE::Core::UI
