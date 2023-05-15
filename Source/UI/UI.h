#pragma once

#include <fmt/core.h>

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"
#include "implot.h"

#include "FontAwesome.h"
#include "Core/Memory.h"
#include "Core/Types.h"
#include "Graphics/Interface/IDescriptorSet.h"

namespace SE::Core::UI
{
    using namespace SE::Graphics;
    
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
        Ref<IDescriptorSet> Handle = nullptr;

        ImageHandle()                      = default;
        ImageHandle( const ImageHandle & ) = default;
    };

    void GetStyleColor(ImGuiCol aColor, math::vec4 *aOut );

} // namespace SE::Core::UI
