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
#include "UI/Components/TextToggleButton.h"
#include "UI/Components/TextOverlay.h"
#include "UI/Form.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/Layouts/ZLayout.h"

#include "DotNet/Class.h"
#include "DotNet/Instance.h"

namespace SE::OtdrEditor
{
    class OtdrWorkspaceWindow : public UIForm
    {
      public:
        OtdrWorkspaceWindow() = default;
        OtdrWorkspaceWindow( OtdrWorkspaceWindow const & ) = default;

        ~OtdrWorkspaceWindow() = default;

        void ConfigureUI();
        void Tick();

      private:
        UIImage mPlayIcon;
        UIImage mPauseIcon;
        UIImageToggleButton mStartOrStopCurrentScript;
        UITextToggleButton mShowLogs;
        UIComboBox          mScriptChooser;
        UIBoxLayout         mTopBarLayout;
        UIBoxLayout         mMainLayout;
        UILabel  mTestLabel0;
        UITextOverlay  mConsoleTextOverlay;

        UIZLayout mWorkspaceLayout;
        UIImage mWorkspaceBackground;

        private:
          std::vector<SE::Core::DotNetClass*> mScripts;

      private:
        bool StartCurrentScript(bool aState);
        Ref<DotNetInstance> mCurrentScript{};
        bool                    mCurrentScriptIsRunning{};

        void ConsoleOut(std::string const& aString);
    };
} // namespace SE::OtdrEditor