#include "Window.h"
#include "Engine/Engine.h"

// std
#include <stdexcept>
// #include <glad/glad.h>

namespace SE::Core
{

    Window::Window( int a_Width, int a_Height, std::string a_Title )
        : mWidth{ a_Width }
        , mHeight{ a_Height }
        , mWindowName{ a_Title }
    {
        InitializeWindow();
    }

    Window::~Window()
    {
        glfwDestroyWindow( mWindow );
        glfwTerminate();
    }

    void Window::InitializeWindow()
    {
        glfwInit();

        // Setup the error callback
        glfwSetErrorCallback( OnGLFWError );

        // Create window without an OpenGL context attached
        glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );

        // Allow window to be resized
        glfwWindowHint( GLFW_RESIZABLE, GLFW_TRUE );

        mWindow = glfwCreateWindow( mWidth, mHeight, mWindowName.c_str(), nullptr, nullptr );
        if( !mWindow ) throw std::runtime_error( "Failed to create window" );

        // Attach a pointer to this class as used data to the underlying GLFW window. This pointer
        // allows us to retrieve our class from callback functions.
        glfwSetWindowUserPointer( mWindow, this );

        // Setup all callback functions
        glfwSetFramebufferSizeCallback( mWindow, OnFramebufferResize );
        glfwSetWindowCloseCallback( mWindow, OnWindowClose );
        glfwSetWindowRefreshCallback( mWindow, OnWindowRefresh );
        glfwSetKeyCallback( mWindow, OnKey );
        glfwSetMouseButtonCallback( mWindow, OnMouseButton );
        glfwSetCursorPosCallback( mWindow, OnCursorPosition );
        glfwSetScrollCallback( mWindow, OnCursorPosition );
        glfwSetCharCallback( mWindow, OnTextInput );

        mLastMousePosition = GetMousePosition();
    }

    math::ivec2 Window::GetMousePosition()
    {
        double lXPos, lYPos;
        glfwGetCursorPos( mWindow, &lXPos, &lYPos );

        return math::vec2{ static_cast<int>( lXPos ), static_cast<int>( lYPos ) };
    }

    math::vec2 Window::GetMainWindowSize()
    {
        int width, height;
        glfwGetWindowSize( mWindow, &width, &height );
        return { static_cast<float>( width ), static_cast<float>( height ) };
    }

    math::ivec2 Window::GetFramebufferSize()
    {
        int width, height;
        glfwGetFramebufferSize( mWindow, &width, &height );
        return { width, height };
    }

    void Window::CreateWindowSurface( VkInstance instance, VkSurfaceKHR *surface )
    {
        if( glfwCreateWindowSurface( instance, mWindow, nullptr, surface ) != VK_SUCCESS )
            throw std::runtime_error( "failed to crrete window surface" );
    }

    void Window::OnFramebufferResize( GLFWwindow *aWindow, int width, int height )
    {
        auto lWindow                    = reinterpret_cast<Window *>( glfwGetWindowUserPointer( aWindow ) );
        lWindow->mFramebufferWasResized = true;
        lWindow->mWidth                 = width;
        lWindow->mHeight                = height;
    }

    void Window::OnGLFWError( int error, const char *description ) { fprintf( stderr, "Error: %s\n", description ); }

    void Window::OnWindowClose( GLFWwindow *aWindow )
    {
        auto lWindow = reinterpret_cast<Window *>( glfwGetWindowUserPointer( aWindow ) );
    }

    void Window::OnWindowRefresh( GLFWwindow *aWindow )
    {
        auto lWindow = reinterpret_cast<Window *>( glfwGetWindowUserPointer( aWindow ) );
    }

    void Window::OnKey( GLFWwindow *aWindow, const int aKey, int aScanCode, const int aAction, const int aModifiers )
    {
        auto lWindow = reinterpret_cast<Window *>( glfwGetWindowUserPointer( aWindow ) );

        UserEvent lUserEvent;

        switch( aAction )
        {
        case GLFW_PRESS: lUserEvent.Type = EventType::KEY_PRESSED; break;
        case GLFW_RELEASE: lUserEvent.Type = EventType::KEY_RELEASED; break;
        case GLFW_REPEAT: lUserEvent.Type = EventType::KEY_REPEAT; break;
        default: lUserEvent.Type = EventType::UNKNOWN;
        };

        lUserEvent.MousePosition = lWindow->mLastMousePosition;
        lUserEvent.MouseDelta    = { 0, 0 };
        lUserEvent.KeyCode       = (Key)aKey;
        lUserEvent.Modifiers     = ModifierFlags( (uint8_t)( (int32_t)aModifiers & 0xff ) );

        lWindow->mEngineLoop->IOEvent( lUserEvent );
    }

    void Window::OnMouseButton( GLFWwindow *aWindow, const int button, const int aAction, const int aModifiers )
    {
        auto lWindow = reinterpret_cast<Window *>( glfwGetWindowUserPointer( aWindow ) );

        UserEvent lUserEvent;

        switch( aAction )
        {
        case GLFW_PRESS: lUserEvent.Type = EventType::MOUSE_BUTTON_PRESSED; break;
        case GLFW_RELEASE: lUserEvent.Type = EventType::MOUSE_BUTTON_RELEASED; break;
        default: lUserEvent.Type = EventType::UNKNOWN;
        };

        lWindow->mLastMousePosition = lWindow->GetMousePosition();

        lUserEvent.MousePosition = lWindow->mLastMousePosition;
        lUserEvent.MouseDelta    = { 0, 0 };
        lUserEvent.KeyCode       = Key::Unknown;
        lUserEvent.Modifiers     = ModifierFlags( (uint8_t)( (int32_t)aModifiers & 0xff ) );
        lWindow->mEngineLoop->IOEvent( lUserEvent );
    }

    void Window::OnCursorPosition( GLFWwindow *aWindow, const double x, const double y )
    {
        auto lWindow = reinterpret_cast<Window *>( glfwGetWindowUserPointer( aWindow ) );

        UserEvent lUserEvent;
        lUserEvent.Type = EventType::MOUSE_MOVE;

        math::ivec2 l_CurrentMousePosition{ static_cast<int32_t>( x ), static_cast<int32_t>( y ) };
        lUserEvent.MousePosition = l_CurrentMousePosition;

        math::ivec2 l_MouseDelta = l_CurrentMousePosition - lWindow->mLastMousePosition;

        lUserEvent.MouseDelta = math::vec2{ static_cast<float>( l_MouseDelta.x ), static_cast<float>( l_MouseDelta.y ) };
        lUserEvent.KeyCode    = Key::Unknown;
        lUserEvent.Modifiers  = ModifierFlags();

        lWindow->mLastMousePosition = l_CurrentMousePosition;
        lWindow->mEngineLoop->IOEvent( lUserEvent );
    }

    void Window::OnMouseScroll( GLFWwindow *aWindow, const double dx, const double dy )
    {
        auto lWindow = reinterpret_cast<Window *>( glfwGetWindowUserPointer( aWindow ) );

        UserEvent lUserEvent;
        lUserEvent.Type = EventType::MOUSE_SCROLL;

        lUserEvent.MousePosition = lWindow->mLastMousePosition;
        lUserEvent.MouseDelta    = math::vec2{ static_cast<float>( dx ), static_cast<float>( dy ) };
        lUserEvent.KeyCode       = Key::Unknown;
        lUserEvent.Modifiers     = ModifierFlags();

        lWindow->mLastMousePosition = lWindow->GetMousePosition();

        lWindow->mEngineLoop->IOEvent( lUserEvent );
    }

    void Window::OnTextInput( GLFWwindow *aWindow, unsigned int codepoint )
    {
        auto lWindow = reinterpret_cast<Window *>( glfwGetWindowUserPointer( aWindow ) );

        UserEvent lUserEvent;
        lUserEvent.MousePosition = lWindow->mLastMousePosition;
        lUserEvent.MouseDelta    = { 0, 0 };
        lUserEvent.KeyCode       = Key::Unknown;
        lUserEvent.Modifiers     = ModifierFlags();
        lWindow->mEngineLoop->IOEvent( lUserEvent );
    }

} // namespace SE::Core
