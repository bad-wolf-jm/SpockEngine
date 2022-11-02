#include "ViewportClient.h"
#include "EngineLoop.h"

// std
#include <stdexcept>
// #include <glad/glad.h>

namespace LTSE::Core
{

    ViewportClient::ViewportClient( int a_Width, int a_Height, std::string a_Title )
        : m_Width{ a_Width }
        , m_Height{ a_Height }
        , m_WindowName{ a_Title }
    {
        InitializeWindow();
    }

    ViewportClient::~ViewportClient()
    {
        glfwDestroyWindow( m_Window );
        glfwTerminate();
    }

    void ViewportClient::InitializeWindow()
    {
        glfwInit();

        // Setup the error callback
        glfwSetErrorCallback( OnGLFWError );

        // Create window without an OpenGL context attached
        glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

        // Allow window to be resized
        glfwWindowHint( GLFW_RESIZABLE, GLFW_TRUE );

        m_Window = glfwCreateWindow( m_Width, m_Height, m_WindowName.c_str(), nullptr, nullptr );
        if( !m_Window ) throw std::runtime_error( "Failed to create window" );

        // Attach a pointer to this class as used data to the underlying GLFW window. This pointer
        // allows us to retrieve our class from callback functions.
        glfwSetWindowUserPointer( m_Window, this );

        // Make the openGL context current;
        // glfwMakeContextCurrent(m_Window);
        // int status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

        // Setup all callback functions
        glfwSetFramebufferSizeCallback( m_Window, OnFramebufferResize );
        glfwSetWindowCloseCallback( m_Window, OnWindowClose );
        glfwSetWindowRefreshCallback( m_Window, OnWindowRefresh );
        glfwSetKeyCallback( m_Window, OnKey );
        glfwSetMouseButtonCallback( m_Window, OnMouseButton );
        glfwSetCursorPosCallback( m_Window, OnCursorPosition );
        glfwSetScrollCallback( m_Window, OnCursorPosition );
        glfwSetCharCallback( m_Window, OnTextInput );

        double xpos, ypos;
        glfwGetCursorPos( m_Window, &xpos, &ypos );
        m_LastMousePosition = math::vec2{ static_cast<float>( xpos ), static_cast<float>( ypos ) };
    }

    math::vec2 ViewportClient::GetMainWindowSize()
    {
        int width, height;
        glfwGetWindowSize( m_Window, &width, &height );
        return { static_cast<float>( width ), static_cast<float>( height ) };
    }

    math::ivec2 ViewportClient::GetFramebufferSize()
    {
        int width, height;
        glfwGetFramebufferSize( m_Window, &width, &height );
        return { width, height };
    }

    void ViewportClient::CreateWindowSurface( VkInstance instance, VkSurfaceKHR *surface )
    {
        if( glfwCreateWindowSurface( instance, m_Window, nullptr, surface ) != VK_SUCCESS )
            throw std::runtime_error( "failed to crrete window surface" );
    }

    void ViewportClient::OnFramebufferResize( GLFWwindow *window, int width, int height )
    {
        auto l_Window                     = reinterpret_cast<ViewportClient *>( glfwGetWindowUserPointer( window ) );
        l_Window->m_FramebufferWasResized = true;
        l_Window->m_Width                 = width;
        l_Window->m_Height                = height;
    }

    void ViewportClient::OnGLFWError( int error, const char *description ) { fprintf( stderr, "Error: %s\n", description ); }

    void ViewportClient::OnWindowClose( GLFWwindow *window )
    {
        auto l_Window = reinterpret_cast<ViewportClient *>( glfwGetWindowUserPointer( window ) );
    }

    void ViewportClient::OnWindowRefresh( GLFWwindow *window )
    {
        auto l_Window = reinterpret_cast<ViewportClient *>( glfwGetWindowUserPointer( window ) );
    }

    void ViewportClient::OnKey( GLFWwindow *window, const int key, int scancode, const int action, const int mods )
    {
        auto      l_Window = reinterpret_cast<ViewportClient *>( glfwGetWindowUserPointer( window ) );
        UserEvent l_UserEvent;

        switch( action )
        {
        case GLFW_PRESS: l_UserEvent.Type = EventType::KEY_PRESSED; break;
        case GLFW_RELEASE: l_UserEvent.Type = EventType::KEY_RELEASED; break;
        case GLFW_REPEAT: l_UserEvent.Type = EventType::KEY_REPEAT; break;
        default: l_UserEvent.Type = EventType::UNKNOWN;
        };

        l_UserEvent.MousePosition = l_Window->m_LastMousePosition;
        l_UserEvent.MouseDelta    = { 0, 0 };
        l_UserEvent.KeyCode       = (Key)key;
        l_UserEvent.Modifiers     = ModifierFlags( (uint8_t)( (int32_t)mods & 0xff ) );
        l_Window->m_EngineLoop->IOEvent( l_UserEvent );
    }

    void ViewportClient::OnMouseButton( GLFWwindow *window, const int button, const int action, const int mods )
    {
        auto      l_Window = reinterpret_cast<ViewportClient *>( glfwGetWindowUserPointer( window ) );
        UserEvent l_UserEvent;

        switch( action )
        {
        case GLFW_PRESS: l_UserEvent.Type = EventType::MOUSE_BUTTON_PRESSED; break;
        case GLFW_RELEASE: l_UserEvent.Type = EventType::MOUSE_BUTTON_RELEASED; break;
        default: l_UserEvent.Type = EventType::UNKNOWN;
        };

        double xpos, ypos;
        glfwGetCursorPos( window, &xpos, &ypos );
        l_Window->m_LastMousePosition = math::vec2{ static_cast<float>( xpos ), static_cast<float>( ypos ) };

        l_UserEvent.MousePosition = l_Window->m_LastMousePosition;
        l_UserEvent.MouseDelta    = { 0, 0 };
        l_UserEvent.KeyCode       = Key::Unknown;
        l_UserEvent.Modifiers     = ModifierFlags( (uint8_t)( (int32_t)mods & 0xff ) );
        l_Window->m_EngineLoop->IOEvent( l_UserEvent );
    }

    void ViewportClient::OnCursorPosition( GLFWwindow *window, const double x, const double y )
    {
        auto l_Window = reinterpret_cast<ViewportClient *>( glfwGetWindowUserPointer( window ) );

        UserEvent l_UserEvent;
        l_UserEvent.Type = EventType::MOUSE_MOVE;

        math::ivec2 l_CurrentMousePosition{ static_cast<int32_t>( x ), static_cast<int32_t>( y ) };
        l_UserEvent.MousePosition = l_CurrentMousePosition;

        math::ivec2 l_MouseDelta = l_CurrentMousePosition - l_Window->m_LastMousePosition;

        l_UserEvent.MouseDelta = math::vec2{ static_cast<float>( l_MouseDelta.x ), static_cast<float>( l_MouseDelta.y ) };
        l_UserEvent.KeyCode    = Key::Unknown;
        l_UserEvent.Modifiers  = ModifierFlags();

        l_Window->m_LastMousePosition = l_CurrentMousePosition;
        l_Window->m_EngineLoop->IOEvent( l_UserEvent );
    }

    void ViewportClient::OnMouseScroll( GLFWwindow *window, const double dx, const double dy )
    {
        auto      l_Window = reinterpret_cast<ViewportClient *>( glfwGetWindowUserPointer( window ) );
        UserEvent l_UserEvent;
        l_UserEvent.Type = EventType::MOUSE_SCROLL;

        l_UserEvent.MousePosition = l_Window->m_LastMousePosition;
        l_UserEvent.MouseDelta    = math::vec2{ static_cast<float>( dx ), static_cast<float>( dy ) };
        l_UserEvent.KeyCode       = Key::Unknown;
        l_UserEvent.Modifiers     = ModifierFlags();

        double xpos, ypos;
        glfwGetCursorPos( window, &xpos, &ypos );
        l_Window->m_LastMousePosition = math::vec2{ static_cast<int32_t>( xpos ), static_cast<int32_t>( ypos ) };

        l_Window->m_EngineLoop->IOEvent( l_UserEvent );
    }

    void ViewportClient::OnTextInput( GLFWwindow *window, unsigned int codepoint )
    {
        auto      l_Window = reinterpret_cast<ViewportClient *>( glfwGetWindowUserPointer( window ) );
        UserEvent l_UserEvent;
        l_UserEvent.MousePosition = l_Window->m_LastMousePosition;
        l_UserEvent.MouseDelta    = { 0, 0 };
        l_UserEvent.KeyCode       = Key::Unknown;
        l_UserEvent.Modifiers     = ModifierFlags();
        l_Window->m_EngineLoop->IOEvent( l_UserEvent );
    }

} // namespace LTSE::Core
