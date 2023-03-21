#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkSampler2D.h"

#include "OtdrScene/OtdrScene.h"

#include "UI/Components/Plot.h"
#include "UI/UI.h"

#include "LinkElementTable.h"

namespace SE::OtdrEditor
{

    using namespace SE::Core;

    class UILinkElementTracePlot : public UIPlot
    {
      public:
        UILinkElementTracePlot()                                 = default;
        UILinkElementTracePlot( UILinkElementTracePlot const & ) = default;

        ~UILinkElementTracePlot() = default;

        void SetData( std::vector<MonoObject *> &lTraceDataVector );
        void SetEventData( sLinkElement const &lEventDataVector, bool aDisplayEventBounds = false, bool aDisplayLsaFit = false );
        void SetEventData( std::vector<sLinkElement> &lEventDataVector );
    };
} // namespace SE::OtdrEditor