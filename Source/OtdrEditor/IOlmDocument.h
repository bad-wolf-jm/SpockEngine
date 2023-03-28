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
#include "UI/Form.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/Layouts/ZLayout.h"

#include "DotNet/Class.h"
#include "DotNet/Instance.h"

#include "LinkElementTable.h"
#include "LinkElementTracePlot.h"

namespace SE::OtdrEditor
{
    class UIIolmDocument : public UIWorkspaceDocument
    {
      public:
        UIIolmDocument() = default;
        UIIolmDocument( fs::path aPath, bool aReanalyse );

        UIIolmDocument( UIIolmDocument const & ) = default;

        ~UIIolmDocument() = default;

      private:
        Ref<DotNetInstance>       mDataInstance = nullptr;
        std::vector<sLinkElement> mLinkElementVector;

      private:
        Ref<UIBoxLayout> mMainLayout;

        Ref<UILinkElementTracePlot> mTracePlot;
        Ref<UILinkElementTable>     mLinkElementTable;
    };
} // namespace SE::OtdrEditor