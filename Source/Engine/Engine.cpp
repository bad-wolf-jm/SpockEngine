#include "Engine.h"

#ifdef APIENTRY
#    undef APIENTRY
#endif
#include <chrono>
#include <shlobj.h>

#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Core
{
    GLFWwindow *Engine::GetMainApplicationWindow() { return mViewportClient->GetGLFWWindow(); }

    int64_t Engine::GetTime()
    {
        auto now    = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>( now );
        return now_ms.time_since_epoch().count();
    }

    int64_t Engine::GetElapsedTime() { return GetTime() - mEngineLoopStartTime; }

    void Engine::PreInit( int argc, char **argv ) {}

    void Engine::Init()
    {
        mGraphicContext         = CreateGraphicContext( 1 );
        mViewportClient         = SE::Core::New<IWindow>( mInitialMainWindowSize.x, mInitialMainWindowSize.y, mApplicationName );
        mSwapChain              = CreateSwapChain( mGraphicContext, mViewportClient );
        mSwapChainRenderContext = CreateRenderContext( mGraphicContext, mSwapChain );
        mViewportClient->SetEngineLoop( this );

        mMainWindowSize  = mViewportClient->GetMainWindowSize();
        mDpiScaling      = math::vec2( 1.0f, 1.0f );
        mFramebufferSize = mViewportClient->GetFramebufferSize();
        mImGUIOverlay =
            New<SE::Core::UIContext>( mViewportClient, mGraphicContext, mSwapChainRenderContext, mImGuiConfigPath, mUIConfiguration );

        mEngineLoopStartTime = GetTime();
        mLastFrameTime       = mEngineLoopStartTime;
    }

    bool Engine::Tick()
    {
        mViewportClient->PollEvents();

        double   time = (double)GetTime();
        Timestep timestep{ static_cast<float>( time - mLastFrameTime ) };

        mLastFrameTime = time;

        if( !mSwapChainRenderContext->BeginRender() ) return true;

        bool lRequestQuit = false;
        mImGUIOverlay->BeginFrame();

        // First run the UI delegate so any state that needs updating for this frame
        // gets updated. If the delegate indicates that we should quit, we return immediately
        if( UIDelegate ) lRequestQuit = UIDelegate( mImGUIOverlay->GetIO() );

        if( lRequestQuit ) return false;

        // Run the update delegate to update the state of the various elements
        // of the simulation.
        if( UpdateDelegate ) UpdateDelegate( timestep );

        // Finally, render the main screen.
        if( RenderDelegate ) RenderDelegate();

        // Render the UI on top of the background
        mImGUIOverlay->EndFrame( mSwapChainRenderContext );

        // // Send the draw commands to the screen.
        mSwapChainRenderContext->EndRender();
        mSwapChainRenderContext->Present();

        mGraphicContext->WaitIdle();

        return true;
    }

    void Engine::IOEvent( UserEvent &a_Event )
    {
        if( IOEventDelegate ) IOEventDelegate( a_Event );
    }

    void Engine::SetInitialWindowPosition( math::ivec2 a_Position ) { mInitialMainWindowPosition = a_Position; }

    void Engine::SetInitialWindowSize( math::ivec2 a_Size ) { mInitialMainWindowSize = a_Size; }

    void Engine::SetImGuiConfigurationFile( std::string a_Path ) { mImGuiConfigPath = a_Path; }

    void Engine::ExecuteMainThreadQueue()
    {
        std::scoped_lock<std::mutex> lock( mMainThreadQueueMutex );

        for( auto &lThunk : mMainThreadQueue ) lThunk();

        mMainThreadQueue.clear();
    }

    void Engine::SubmitToMainThread( const std::function<void()> &aThunk )
    {
        std::scoped_lock<std::mutex> lock( mMainThreadQueueMutex );

        mMainThreadQueue.emplace_back( aThunk );
    }

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

    std::unique_ptr<Engine> Engine::mUniqueInstance = nullptr;

    void Engine::Initialize( math::ivec2 aInitialMainWindowSize, math::ivec2 aInitialMainWindowPosition, fs::path aImGuiConfigPath,
                             UIConfiguration const &aUIConfiguration )
    {
        IWindow::InitializeWindowingBackend();

        if( mUniqueInstance ) return;

        mUniqueInstance = std::make_unique<Engine>();

        mUniqueInstance->mInitialMainWindowSize     = aInitialMainWindowSize;
        mUniqueInstance->mInitialMainWindowPosition = aInitialMainWindowPosition;
        mUniqueInstance->mImGuiConfigPath           = aImGuiConfigPath.string();
        mUniqueInstance->mUIConfiguration           = aUIConfiguration;
        mUniqueInstance->Init();
    }

    void Engine::Shutdown() { IWindow::ShutdownWindowingBackend(); }

    std::unique_ptr<Engine> &Engine::GetInstance() { return mUniqueInstance; };

} // namespace SE::Core