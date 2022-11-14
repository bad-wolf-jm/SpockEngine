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

    GLFWwindow *EngineLoop::GetMainApplicationWindow() { return m_ViewportClient->GetGLFWWindow(); }

    int64_t EngineLoop::GetTime()
    {
        auto now    = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>( now );
        return now_ms.time_since_epoch().count();
    }

    int64_t EngineLoop::GetElapsedTime() { return GetTime() - m_EngineLoopStartTime; }

    void EngineLoop::PreInit( int argc, char **argv )
    {
        CHAR profilePath[MAX_PATH];
        HRESULT result = SHGetFolderPathA( NULL, CSIDL_PROFILE, NULL, 0, profilePath );
        if( SUCCEEDED( result ) )
        {
            m_UserHomeFolder = std::string( profilePath );
        }

        CHAR appData[MAX_PATH];
        result = SHGetFolderPathA( NULL, CSIDL_LOCAL_APPDATA, NULL, 0, appData );
        if( SUCCEEDED( result ) )
        {
            m_LocalConfigFolder = std::string( appData );
        }

        LTSE::Graphics::OptixDeviceContextObject::Initialize();

        m_EngineLoopStartTime = GetTime();
        m_LastFrameTime       = m_EngineLoopStartTime;
    }

    void EngineLoop::Init()
    {
        mGraphicContext = LTSE::Graphics::GraphicContext( m_InitialMainWindowSize.x, m_InitialMainWindowSize.y, 1, m_ApplicationName );

        SwapChainRenderTargetDescription l_SwapChainSettings{ 1 };
        m_SwapChainRenderer = LTSE::Core::New<SwapChainRenderTarget>( mGraphicContext, l_SwapChainSettings );
        m_RenderContext     = LTSE::Graphics::RenderContext( mGraphicContext, m_SwapChainRenderer );

        m_ViewportClient = mGraphicContext.GetViewportClient();

        m_ViewportClient->SetEngineLoop( this );

        m_MainWindowSize  = m_ViewportClient->GetMainWindowSize();
        m_DpiScaling      = math::vec2( 1.0f, 1.0f );
        m_FramebufferSize = m_ViewportClient->GetFramebufferSize();
        mImGUIOverlay    = New<LTSE::Core::UIContext>( m_ViewportClient, mGraphicContext, m_RenderContext, mImGuiConfigPath );
    }

    void EngineLoop::Shutdown() {}

    bool EngineLoop::Tick()
    {
        m_ViewportClient->PollEvents();

        double time = (double)GetTime();
        Timestep timestep{ static_cast<float>( time - m_LastFrameTime ) };

        m_LastFrameTime = time;

        if( !m_RenderContext.BeginRender() )
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
        mImGUIOverlay->EndFrame( m_RenderContext );

        // Send the draw commands to the screen.
        m_RenderContext.EndRender();
        m_RenderContext.Present();

        mGraphicContext.WaitIdle();
        return true;
    }

    void EngineLoop::IOEvent( UserEvent &a_Event )
    {
        if( IOEventDelegate )
            IOEventDelegate( a_Event );
    }

    void EngineLoop::SetInitialWindowPosition( math::ivec2 a_Position ) { m_InitialMainWindowPosition = a_Position; }

    void EngineLoop::SetInitialWindowSize( math::ivec2 a_Size ) { m_InitialMainWindowSize = a_Size; }

    void EngineLoop::SetImGuiConfigurationFile( std::string a_Path ) { mImGuiConfigPath = a_Path; }

    math::ivec2 EngineLoop::GetWindowPosition()
    {
        int x, y;
        glfwGetWindowPos( m_ViewportClient->GetGLFWWindow(), &x, &y );
        return { x, y };
    }

    math::ivec2 EngineLoop::GetWindowSize()
    {
        int w, h;
        glfwGetWindowSize( m_ViewportClient->GetGLFWWindow(), &w, &h );
        return { w, h };
    }

    std::string EngineLoop::GetImGuiConfigurationFile() { return mImGuiConfigPath; }

} // namespace LTSE::Core