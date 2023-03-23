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
        if( mApplicationInstance )
        {
            float lTs = ts.GetMilliseconds();
            mApplicationInstance->CallMethod( "Update", &lTs );
        }
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

        if( mApplicationInstance )
        {
            float lTs = 0.0f;
            mApplicationInstance->CallMethod( "UpdateUI", &lTs );
        }

        return lRequestQuit;
    }

    void BaseOtdrApplication::Init()
    {
        mEditorWindow =
            OtdrWindow( SE::Core::Engine::GetInstance()->GetGraphicContext(), SE::Core::Engine::GetInstance()->UIContext() );
        mEditorWindow.ConfigureUI();
        mEditorWindow.ApplicationIcon = ICON_FA_CODEPEN;

        static auto &lApplicationType    = MonoRuntime::GetClassType( "SpockEngine.SEApplication" );
        auto &lApplicationClasses = lApplicationType.DerivedClasses();
        if( lApplicationClasses.size() > 0 )
        {
            mApplicationInstance = lApplicationClasses[0]->Instantiate();
            mApplicationInstance->CallMethod( "Initialize" );
        }
    }

    void BaseOtdrApplication::Shutdown()
    {
        if( mApplicationInstance ) mApplicationInstance->CallMethod( "Shutdown" );
    }
} // namespace SE::OtdrEditor
