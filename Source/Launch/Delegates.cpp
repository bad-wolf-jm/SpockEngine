#include "Delegates.h"

#include "Core/File.h"
#include "yaml-cpp/yaml.h"
#include <fstream>

#include <direct.h>
#include <iostream>
#include <limits.h>
#include <string>

#include "UI/UI.h"
// #include "UI/Widgets.h"

#include "DotNet/Runtime.h"

namespace SE::OtdrEditor
{

    using namespace SE::Core;

    void Application::Update( Timestep ts )
    {
        mEditorWindow.Update( ts );
        if( mManaged ) mManaged->Update( ts.GetMilliseconds() );
    }

    bool Application::RenderUI( ImGuiIO &io )
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

        if( mManaged )
        {
            float lTs = 0.0f;
            mManaged->UpdateUI( lTs );
        }

        return lRequestQuit;
    }

    void Application::Init()
    {
        mEditorWindow =
            MainWindow( SE::Core::Engine::GetInstance()->GetGraphicContext(), SE::Core::Engine::GetInstance()->UIContext() );
        mEditorWindow.ApplicationIcon = ICON_FA_CODEPEN;
    }

    void Application::Init( fs::path aConfigurationPath )
    {
        Init();

        if( mManaged )
        {
            mManaged->Configure( aConfigurationPath.string() );
            mEditorWindow.mManaged = mManaged;
        }
    }

    void Application::Shutdown( fs::path aConfigurationPath )
    {
        if( mManaged ) mManaged->Teardown( aConfigurationPath.string() );
    }
} // namespace SE::OtdrEditor
