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

        mStartOrStopCurrentScript.SetInactiveImage( mPlayIcon );
        mStartOrStopCurrentScript.SetActiveImage( mPauseIcon );
        mStartOrStopCurrentScript.OnChange( [&]( bool ) { return true; } );

        mShowLogs = UITextToggleButton("Logs");

        mTopBarLayout = UIBoxLayout( eBoxLayoutOrientation::HORIZONTAL );
        mTopBarLayout.Add( &mStartOrStopCurrentScript, 45.0f, false, true );
        mTopBarLayout.Add( &mScriptChooser, 145.0f, false, true );
        mTopBarLayout.Add( nullptr, true, true );
        mTopBarLayout.Add( &mShowLogs, false, true );

        mMainLayout = UIBoxLayout( eBoxLayoutOrientation::VERTICAL );
        mMainLayout.Add( &mTopBarLayout, 45.0f, false, true );

        mTestLabel0 = UILabel( "SCRIPT GUI GOES HERE" );
        mMainLayout.Add( &mTestLabel0, true, true );

        SetTitle( "WORKSPACE" );
        SetContent( &mMainLayout );
    }

} // namespace SE::OtdrEditor