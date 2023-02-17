#include "BaseOtdrApplication.h"

#include "Core/File.h"
#include "yaml-cpp/yaml.h"
#include <fstream>

#include <direct.h>
#include <iostream>
#include <limits.h>
#include <string>

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Mono/MonoRuntime.h"

namespace SE::OtdrEditor
{

    using namespace SE::Core;

    void BaseOtdrApplication::Update( Timestep ts )
    {
        mEditorWindow.Update( ts );
    }

    bool BaseOtdrApplication::RenderUI( ImGuiIO &io )
    {
        bool lRequestQuit = false;

        lRequestQuit = mEditorWindow.Display();

        auto lWorkspaceAreaSize = mEditorWindow.GetWorkspaceAreaSize();
        if( ( mViewportWidth != lWorkspaceAreaSize.x ) || ( mViewportHeight != lWorkspaceAreaSize.y ) )
        {
            mViewportWidth         = lWorkspaceAreaSize.x;
            mViewportHeight        = lWorkspaceAreaSize.y;
            mShouldRebuildViewport = true;
        }

        return lRequestQuit;
    }

    void BaseOtdrApplication::Init()
    {
        mEditorWindow =
            OtdrWindow( SE::Core::Engine::GetInstance()->GetGraphicContext(), SE::Core::Engine::GetInstance()->UIContext() );
        mEditorWindow.ConfigureUI();
        mEditorWindow.ApplicationIcon = ICON_FA_CODEPEN;

        mWorld = New<OtdrScene>();

        mEditorWindow.mWorld       = mWorld;
        mEditorWindow.mActiveWorld = mWorld;
    }
} // namespace SE::Editor
