#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkSampler2D.h"

#include "OtdrScene/OtdrScene.h"

#include "UI/Components/Label.h"
#include "UI/Components/Plot.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/UI.h"

#include "LinkElementTable.h"

namespace SE::OtdrEditor
{

    using namespace SE::Core;

    class UILinkElementTracePlot : public UIBoxLayout
    {
      public:
        UILinkElementTracePlot();

        UILinkElementTracePlot( UILinkElementTracePlot const & ) = default;

        ~UILinkElementTracePlot() = default;

        void SetTitle(std::string aTitle);
        void SetSubTitle(std::string aTitle);

        void SetData( std::vector<MonoObject *> &lTraceDataVector );
        void SetEventData( sLinkElement const &lEventDataVector, bool aDisplayEventBounds = false, bool aDisplayLsaFit = false,
                           bool aAdjustAxisScale = false );
        void SetEventData( std::vector<sLinkElement> &lEventDataVector );

        void Clear();
      private:
        Ref<UILabel> mTitle;
        Ref<UILabel> mSubTitle;
        Ref<UIPlot>  mPlotArea;
    };
} // namespace SE::OtdrEditor