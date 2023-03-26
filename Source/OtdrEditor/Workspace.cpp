#include "Workspace.h"

#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>

#include "Core/Profiling/BlockTimer.h"

#include "Core/File.h"
#include "Core/Logging.h"

#include "Mono/MonoRuntime.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;

    void OtdrWorkspaceWindow::ConfigureUI()
    {
        mPlayIcon = UIImage( "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Play.png", math::vec2{ 30, 30 } );
        mPlayIcon.SetTintColor( math::vec4{ 0.0f, 1.0f, 0.0f, 0.8f } );

        mPauseIcon = UIImage( "C:\\GitLab\\SpockEngine\\Saved\\Resources\\Icons\\Pause.png", math::vec2{ 30, 30 } );
        mPauseIcon.SetTintColor( math::vec4{ 1.0f, .2f, 0.0f, 0.8f } );

        mStartOrStopCurrentScript.SetInactiveImage( &mPlayIcon );
        mStartOrStopCurrentScript.SetActiveImage( &mPauseIcon );
        mStartOrStopCurrentScript.OnChange( [&]( bool ) { return StartCurrentScript( true ); } );

        mShowLogs = UITextToggleButton( "Logs" );
        mShowLogs.SetActive( true );
        mShowLogs.OnChange( [&]( bool aValue ) { return mConsoleTextOverlay.mIsVisible = !aValue; } );

        mTopBarLayout = UIBoxLayout( eBoxLayoutOrientation::HORIZONTAL );
        mTopBarLayout.Add( &mStartOrStopCurrentScript, 40.0f, false, true );
        mTopBarLayout.Add( &mScriptChooser, 350.0f, false, true );
        mTopBarLayout.Add( nullptr, true, true );
        mTopBarLayout.Add( &mShowLogs, false, true );

        mMainLayout = UIBoxLayout( eBoxLayoutOrientation::VERTICAL );
        mMainLayout.SetItemSpacing( 15.0f );
        mMainLayout.Add( &mTopBarLayout, 40.0f, false, true );
        mMainLayout.Add( &mWorkspaceLayout, true, true );

        mWorkspaceBackground = UIImage(
            "C:\\GitLab\\SpockEngine\\Programs\\TestOtdrProject\\Resources\\Global-Fiber-Optic-Network-JBL-Communications.jpg",
            math::vec2{ 1300, 765 } );
        mWorkspaceBackground.mIsVisible = true;

        mTestLabel0 = UILabel( "SCRIPT GUI GOES HERE" );
        mWorkspaceLayout.Add( &mWorkspaceBackground, true, true );
        mWorkspaceLayout.Add( &mTestLabel0, true, true );
        mWorkspaceLayout.Add( &mConsoleTextOverlay, math::vec2{ 1.0f, 0.5f }, math::vec2{ 0.0f, 0.5f }, false, true );

        mConsoleTextOverlay.mIsVisible = mShowLogs.IsActive();

        SetTitle( "WORKSPACE" );
        SetContent( &mMainLayout );

        MonoRuntime::OnConsoleOut( std::bind( &OtdrWorkspaceWindow::ConsoleOut, this, std::placeholders::_1 ) );
    }

    void OtdrWorkspaceWindow::ConsoleOut( std::string const &aString ) { mConsoleTextOverlay.AddText( aString ); }

    void OtdrWorkspaceWindow::Tick()
    {
        if( mCurrentScript && mCurrentScriptIsRunning )
        {
            Timestep aTs;
            mCurrentScript->CallMethod( "Tick", &aTs );
        }

        std::vector<std::string> lScriptNames;
        mScripts.clear();

        auto lScriptBaseClass = MonoRuntime::GetClassType( "SpockEngine.Script" );

        for( auto const &lScriptClass : lScriptBaseClass.DerivedClasses() )
        {
            lScriptNames.push_back( lScriptClass->FullName() );
            mScripts.push_back( lScriptClass );
        }

        mScriptChooser.SetItemList( lScriptNames );
    }

    bool OtdrWorkspaceWindow::StartCurrentScript( bool aState )
    {
        if( mCurrentScriptIsRunning )
        {
            mCurrentScript->CallMethod( "EndScenario" );
            mCurrentScriptIsRunning = false;
            mCurrentScript          = nullptr;
        }
        else
        {
            mCurrentScript = mScripts[mScriptChooser.Current()]->Instantiate();
            mCurrentScript->CallMethod( "BeginScenario" );
            mCurrentScriptIsRunning = true;
        }

        return mCurrentScriptIsRunning;
    }

} // namespace SE::OtdrEditor