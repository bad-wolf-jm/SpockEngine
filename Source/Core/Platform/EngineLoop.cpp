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

    GLFWwindow *Engine::GetMainApplicationWindow() { return mViewportClient->GetGLFWWindow(); }

    int64_t Engine::GetTime()
    {
        auto now    = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>( now );
        return now_ms.time_since_epoch().count();
    }

    int64_t Engine::GetElapsedTime() { return GetTime() - mEngineLoopStartTime; }

    void Engine::PreInit( int argc, char **argv )
    {
        mEngineLoopStartTime = GetTime();
        mLastFrameTime       = mEngineLoopStartTime;
    }

    void Engine::Init()
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

    void Engine::Shutdown() {}

    bool Engine::Tick()
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
        if( UIDelegate )
            requestQuit = UIDelegate( mImGUIOverlay->GetIO() );

        if( requestQuit )
            return false;

        // Run the update delegate to update the state of the various elements
        // of the simulation.
        if( UpdateDelegate )
            UpdateDelegate( timestep );

        // Finally, render the main screen.
        if( RenderDelegate )
            RenderDelegate();

        // Render the UI on top of the background
        mImGUIOverlay->EndFrame( mRenderContext );

        // Send the draw commands to the screen.
        mRenderContext.EndRender();
        mRenderContext.Present();

        mGraphicContext.WaitIdle();
        return true;
    }

    void Engine::IOEvent( UserEvent &a_Event )
    {
        if( IOEventDelegate )
            IOEventDelegate( a_Event );
    }

    void Engine::SetInitialWindowPosition( math::ivec2 a_Position ) { mInitialMainWindowPosition = a_Position; }

    void Engine::SetInitialWindowSize( math::ivec2 a_Size ) { mInitialMainWindowSize = a_Size; }

    void Engine::SetImGuiConfigurationFile( std::string a_Path ) { mImGuiConfigPath = a_Path; }

    math::ivec2 Engine::GetWindowPosition()
    {
        int x, y;
        glfwGetWindowPos( mViewportClient->GetGLFWWindow(), &x, &y );
        return { x, y };
    }

    math::ivec2 Engine::GetWindowSize()
    {
        int w, h;
        glfwGetWindowSize( mViewportClient->GetGLFWWindow(), &w, &h );
        return { w, h };
    }

    std::string Engine::GetImGuiConfigurationFile() { return mImGuiConfigPath; }

} // namespace LTSE::Core