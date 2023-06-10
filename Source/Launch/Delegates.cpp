#include "BaseOtdrApplication.h"

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

    void BaseOtdrApplication::Update( Timestep ts )
    {
        mEditorWindow.Update( ts );
        if( mManaged ) mManaged->Update( ts.GetMilliseconds() );
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

        if( mManaged )
        {
            float lTs = 0.0f;
            mManaged->UpdateUI( lTs );
        }

        return lRequestQuit;
    }

    void BaseOtdrApplication::Init()
    {
        mEditorWindow =
            OtdrWindow( SE::Core::Engine::GetInstance()->GetGraphicContext(), SE::Core::Engine::GetInstance()->UIContext() );
        mEditorWindow.ApplicationIcon = ICON_FA_CODEPEN;
    }

    void BaseOtdrApplication::Init( std::string aAppClass, fs::path aConfigurationPath )
    {
        Init();

        // static auto &lApplicationType = DotNetRuntime::GetClassType( aAppClass );

        if( mManaged )
        {
            mManaged->Configure()
            mEditorWindow.mManaged = mManaged;
        }
        else
        {
            SE::Logging::Info( "Could not load application: class {} does not exist", aAppClass );
        }
    }

    void BaseOtdrApplication::Shutdown( fs::path aConfigurationPath )
    {
        // auto lConfigurationPath = DotNetRuntime::NewString( aConfigurationPath.string() );
        if( mManaged ) mApplicationInstance->Teardown();
    }
} // namespace SE::OtdrEditor
