#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "UI/Components/Button.h"
#include "UI/Components/ComboBox.h"
#include "UI/Components/Image.h"
#include "UI/Components/ImageToggleButton.h"
#include "UI/Components/TextOverlay.h"
#include "UI/Components/TextToggleButton.h"
#include "UI/Components/Workspace.h"
#include "UI/Components/PropertyValue.h"
#include "UI/Form.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/Layouts/ZLayout.h"

#include "DotNet/Class.h"
#include "DotNet/Instance.h"

#include "EventOverview.h"
#include "LinkElementTable.h"
#include "LinkElementTracePlot.h"

namespace SE::OtdrEditor
{
    class UIIolmDiffDocument : public UIWorkspaceDocument
    {
      public:
        UIIolmDiffDocument() = default;
        UIIolmDiffDocument( fs::path aPath, bool aReanalyse );

        UIIolmDiffDocument( UIIolmDiffDocument const & ) = default;

        ~UIIolmDiffDocument() = default;

      private:
        Ref<DotNetInstance>       mDataInstance = nullptr;
        std::vector<sLinkElement> mLinkElementVector0;
        std::vector<sLinkElement> mLinkElementVector1;

      private:
        Ref<UIBoxLayout> mTopLayout;

        Ref<UIPropertyValue> mLaunchFiberLength;
        Ref<UIPropertyValue> mReceiveFiberLength;
        Ref<UIPropertyValue> mLinkLength;
        Ref<UIPropertyValue> mFiberCode;

        Ref<UIBoxLayout> mEventLayout;
        Ref<UIBoxLayout> mMainLayout;

        Ref<UILinkElementTracePlot> mTracePlot;
        Ref<UILinkElementTable>     mLinkElementTable0;
        Ref<UILinkElementTable>     mLinkElementTable1;
    };
} // namespace SE::OtdrEditor