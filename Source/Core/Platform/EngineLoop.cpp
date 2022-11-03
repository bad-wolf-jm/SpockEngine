#include "EngineLoop.h"

#ifdef APIENTRY
#    undef APIENTRY
#endif
#include <chrono>
#include <shlobj.h>

#include "Core/Logging.h"
#include "Core/Memory.h"
#ifndef g_optixFunctionTable
#    include <optix_function_table_definition.h>
#endif

// #include "Socket.h"

namespace LTSE::Core
{

    GLFWwindow *EngineLoop::GetMainApplicationWindow() { return mViewportClient->GetGLFWWindow(); }

    int64_t EngineLoop::GetTime()
    {
        auto now    = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>( now );
        return now_ms.time_since_epoch().count();
    }

    int64_t EngineLoop::GetElapsedTime() { return GetTime() - mEngineLoopStartTime; }

    void EngineLoop::PreInit( int argc, char **argv )
    {
        CHAR    profilePath[MAX_PATH];
        HRESULT result = SHGetFolderPathA( NULL, CSIDL_PROFILE, NULL, 0, profilePath );
        if( SUCCEEDED( result ) )
        {
            mUserHomeFolder = std::string( profilePath );
        }

        CHAR appData[MAX_PATH];
        result = SHGetFolderPathA( NULL, CSIDL_LOCAL_APPDATA, NULL, 0, appData );
        if( SUCCEEDED( result ) )
        {
            mLocalConfigFolder = std::string( appData );
        }

        LTSE::Graphics::OptixDeviceContextObject::Initialize();

        mEngineLoopStartTime = GetTime();
        mLastFrameTime       = mEngineLoopStartTime;
    }

    void EngineLoop::Init()
    {
        mGraphicContext = LTSE::Graphics::GraphicContext( mInitialMainWindowSize.x, mInitialMainWindowSize.y, 4, mApplicationName );

        SwapChainRenderTargetDescription l_SwapChainSettings{ 4 };
        mSwapChainRenderer = LTSE::Core::New<SwapChainRenderTarget>( mGraphicContext, l_SwapChainSettings );
        mRenderContext     = LTSE::Graphics::RenderContext( mGraphicContext, mSwapChainRenderer );

        mViewportClient = mGraphicContext.GetViewportClient();

        mViewportClient->SetEngineLoop( this );

        mMainWindowSize  = mViewportClient->GetMainWindowSize();
        mDpiScaling      = math::vec2( 1.0f, 1.0f );
        mFramebufferSize = mViewportClient->GetFramebufferSize();
        mImGUIOverlay    = New<LTSE::Core::UIContext>( mViewportClient, mGraphicContext, mRenderContext, mImGuiConfigPath );
    }

    void EngineLoop::Shutdown() {}

    bool EngineLoop::Tick()
    {
        mViewportClient->PollEvents();

        double time = (double)GetTime();
        Timestep timestep{ static_cast<float>( time - mLastFrameTime ) };

        mLastFrameTime = time;

        if( !mRenderContext.BeginRender() )
            return true;

        bool requestQuit = false;
        mImGUIOverlay->BeginFrame();

        // First run the UI delegate so any state that needs updating for this frame
        // gets updated. If the delegate indicates that we should quit, we return immediately
        if( UIDelegate ) requestQuit = UIDelegate( mImGUIOverlay->GetIO() );

        if( requestQuit ) return false;

        // Run the update delegate to update the state of the various elements
        // of the simulation.
        if( UpdateDelegate ) UpdateDelegate( timestep );

        // Finally, render the main screen.
        if( RenderDelegate ) RenderDelegate();

        // Render the UI on top of the background
        mImGUIOverlay->EndFrame( mRenderContext );

        // Send the draw commands to the screen.
        mRenderContext.EndRender();
        mRenderContext.Present();

        mGraphicContext.WaitIdle();
        return true;
    }

    void EngineLoop::IOEvent( UserEvent &a_Event )
    {
        if( IOEventDelegate ) IOEventDelegate( a_Event );
    }

    void EngineLoop::SetInitialWindowPosition( math::ivec2 a_Position ) { mInitialMainWindowPosition = a_Position; }

    void EngineLoop::SetInitialWindowSize( math::ivec2 a_Size ) { mInitialMainWindowSize = a_Size; }

    void EngineLoop::SetImGuiConfigurationFile( std::string a_Path ) { mImGuiConfigPath = a_Path; }

    math::ivec2 EngineLoop::GetWindowPosition()
    {
        int x, y;
        glfwGetWindowPos( mViewportClient->GetGLFWWindow(), &x, &y );
        return { x, y };
    }

    math::ivec2 EngineLoop::GetWindowSize()
    {
        int w, h;
        glfwGetWindowSize( mViewportClient->GetGLFWWindow(), &w, &h );
        return { w, h };
    }

    std::string EngineLoop::GetImGuiConfigurationFile() { return mImGuiConfigPath; }

} // namespace LTSE::Core