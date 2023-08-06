#pragma once

#include <fmt/core.h>

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"
#include "implot.h"

#include "Core/Memory.h"
#include "Core/Types.h"
#include "FontAwesome.h"
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
        ref_t<IDescriptorSet> Handle = nullptr;

        ImageHandle()                      = default;
        ImageHandle( const ImageHandle & ) = default;
    };
} // namespace SE::Core::UI
