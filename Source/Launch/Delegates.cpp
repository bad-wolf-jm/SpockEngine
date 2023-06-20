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

namespace SE::OtdrEditor
{

    using namespace SE::Core;

    Application::Application( UpdateFn aUpdateDelegate, RenderSceneFn aRenderDelegate, RenderUIFn aRenderUIDelegate )
        : mUpdateDelegate{ aUpdateDelegate }
        , mRenderDelegate{ aRenderDelegate }
        , mRenderUIDelegate{ aRenderUIDelegate }
    {
    }

    void Application::Update( Timestep ts )
    {
        mEditorWindow.Update( ts );

        if( mUpdateDelegate ) mUpdateDelegate( ts.GetMilliseconds() );
    }

    void Application::RenderScene()
    {
        if( mRenderDelegate ) mRenderDelegate();
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

        if( mRenderUIDelegate )
        {
            float lTs = 0.0f;
            lRequestQuit |= mRenderUIDelegate( lTs );
        }

        return lRequestQuit;
    }

    // void Application::Init()
    // {
    //     mEditorWindow =
    //         MainWindow( SE::Core::Engine::GetInstance()->GetGraphicContext(), SE::Core::Engine::GetInstance()->UIContext() );
    //     mEditorWindow.ApplicationIcon = ICON_FA_CODEPEN;
    // }

    // void Application::Init( path_t aConfigurationPath )
    // {
    //     Init();

    //     if( mManaged )
    //     {
    //         mManaged->Configure( aConfigurationPath.string() );
    //         mEditorWindow.mManaged = mManaged;
    //     }
    // }

    // void Application::Shutdown( path_t aConfigurationPath )
    // {
    //     if( mManaged ) mManaged->Teardown( aConfigurationPath.string() );
    // }
} // namespace SE::OtdrEditor
