#pragma once

#include <string>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include "Core/Math/Types.h"
#include "Core/Types.h"

// #include <entt/entt.hpp>

namespace SE::Core
{

    /** @brief */
    enum class MouseButton : size_t
    {

    };

    /** @brief */
    enum class Key : int32_t
    {
        Unknown = GLFW_KEY_UNKNOWN, /**< Unknown key */

        LeftShift  = GLFW_KEY_LEFT_SHIFT,
        RightShift = GLFW_KEY_RIGHT_SHIFT,

        LeftCtrl  = GLFW_KEY_LEFT_CONTROL,
        RightCtrl = GLFW_KEY_RIGHT_CONTROL,

        LeftAlt  = GLFW_KEY_LEFT_ALT,
        RightAlt = GLFW_KEY_RIGHT_ALT,

        LeftSuper  = GLFW_KEY_LEFT_SUPER,
        RightSuper = GLFW_KEY_RIGHT_SUPER,

        Enter = GLFW_KEY_ENTER,  /**< Enter  */
        Esc   = GLFW_KEY_ESCAPE, /**< Escape */

        Up        = GLFW_KEY_UP,        /**!< Up arrow    */
        Down      = GLFW_KEY_DOWN,      /**!< Down arrow  */
        Left      = GLFW_KEY_LEFT,      /**!< Left arrow  */
        Right     = GLFW_KEY_RIGHT,     /**!< Right arrow */
        Home      = GLFW_KEY_HOME,      /**!< Home        */
        End       = GLFW_KEY_END,       /**!< End         */
        PageUp    = GLFW_KEY_PAGE_UP,   /**!< Page up     */
        PageDown  = GLFW_KEY_PAGE_DOWN, /**!< Page down   */
        Backspace = GLFW_KEY_BACKSPACE, /**!< Backspace   */
        Insert    = GLFW_KEY_INSERT,    /**!< Insert      */
        Delete    = GLFW_KEY_DELETE,    /**!< Delete      */

        F1  = GLFW_KEY_F1,  /**!< F1  */
        F2  = GLFW_KEY_F2,  /**!< F2  */
        F3  = GLFW_KEY_F3,  /**!< F3  */
        F4  = GLFW_KEY_F4,  /**!< F4  */
        F5  = GLFW_KEY_F5,  /**!< F5  */
        F6  = GLFW_KEY_F6,  /**!< F6  */
        F7  = GLFW_KEY_F7,  /**!< F7  */
        F8  = GLFW_KEY_F8,  /**!< F8  */
        F9  = GLFW_KEY_F9,  /**!< F9  */
        F10 = GLFW_KEY_F10, /**!< F10 */
        F11 = GLFW_KEY_F11, /**!< F11 */
        F12 = GLFW_KEY_F12, /**!< F12 */

        Space = GLFW_KEY_SPACE, /**!< Space */
        Tab   = GLFW_KEY_TAB,   /**!< Tab   */

        Quote = GLFW_KEY_APOSTROPHE,

        Comma  = GLFW_KEY_COMMA,  /**!< Comma  */
        Period = GLFW_KEY_PERIOD, /**!< Period */
        Minus  = GLFW_KEY_MINUS,  /**!< Minus  */

        Plus  = '+',            /**!< Plus  */
        Slash = GLFW_KEY_SLASH, /**!< Slash */

        Percent   = '%',                /**!< Percent  */
        Semicolon = GLFW_KEY_SEMICOLON, /**!< Semicolon */

        Equal = GLFW_KEY_EQUAL, /**!< Equal */

        LeftBracket  = GLFW_KEY_LEFT_BRACKET,  /**!< Left bracket */
        RightBracket = GLFW_KEY_RIGHT_BRACKET, /**!< Right bracket */
        Backslash    = GLFW_KEY_BACKSLASH,     /**!< Backslach */
        Backquote    = GLFW_KEY_GRAVE_ACCENT,  /**!< Grave accent */

        World1 = GLFW_KEY_WORLD_1,
        World2 = GLFW_KEY_WORLD_2,

        Zero  = GLFW_KEY_0, /**!< Zero  */
        One   = GLFW_KEY_1, /**!< One   */
        Two   = GLFW_KEY_2, /**!< Two   */
        Three = GLFW_KEY_3, /**!< Three */
        Four  = GLFW_KEY_4, /**!< Four  */
        Five  = GLFW_KEY_5, /**!< Five  */
        Six   = GLFW_KEY_6, /**!< Six   */
        Seven = GLFW_KEY_7, /**!< Seven */
        Eight = GLFW_KEY_8, /**!< Eight */
        Nine  = GLFW_KEY_9, /**!< Nine  */

        A = GLFW_KEY_A, /**!< Letter A */
        B = GLFW_KEY_B, /**!< Letter B */
        C = GLFW_KEY_C, /**!< Letter C */
        D = GLFW_KEY_D, /**!< Letter D */
        E = GLFW_KEY_E, /**!< Letter E */
        F = GLFW_KEY_F, /**!< Letter F */
        G = GLFW_KEY_G, /**!< Letter G */
        H = GLFW_KEY_H, /**!< Letter H */
        I = GLFW_KEY_I, /**!< Letter I */
        J = GLFW_KEY_J, /**!< Letter J */
        K = GLFW_KEY_K, /**!< Letter K */
        L = GLFW_KEY_L, /**!< Letter L */
        M = GLFW_KEY_M, /**!< Letter M */
        N = GLFW_KEY_N, /**!< Letter N */
        O = GLFW_KEY_O, /**!< Letter O */
        P = GLFW_KEY_P, /**!< Letter P */
        Q = GLFW_KEY_Q, /**!< Letter Q */
        R = GLFW_KEY_R, /**!< Letter R */
        S = GLFW_KEY_S, /**!< Letter S */
        T = GLFW_KEY_T, /**!< Letter T */
        U = GLFW_KEY_U, /**!< Letter U */
        V = GLFW_KEY_V, /**!< Letter V */
        W = GLFW_KEY_W, /**!< Letter W */
        X = GLFW_KEY_X, /**!< Letter X */
        Y = GLFW_KEY_Y, /**!< Letter Y */
        Z = GLFW_KEY_Z, /**!< Letter Z */

        CapsLock    = GLFW_KEY_CAPS_LOCK,    /**!< Caps lock    */
        ScrollLock  = GLFW_KEY_SCROLL_LOCK,  /**!< Scroll lock  */
        NumLock     = GLFW_KEY_NUM_LOCK,     /**!< Num lock     */
        PrintScreen = GLFW_KEY_PRINT_SCREEN, /**!< Print screen */
        Pause       = GLFW_KEY_PAUSE,        /**!< Pause        */
        Menu        = GLFW_KEY_MENU,         /**!< Menu         */

        NumZero     = GLFW_KEY_KP_0,        /**!< Numpad zero     */
        NumOne      = GLFW_KEY_KP_1,        /**!< Numpad one      */
        NumTwo      = GLFW_KEY_KP_2,        /**!< Numpad two      */
        NumThree    = GLFW_KEY_KP_3,        /**!< Numpad three    */
        NumFour     = GLFW_KEY_KP_4,        /**!< Numpad four     */
        NumFive     = GLFW_KEY_KP_5,        /**!< Numpad five     */
        NumSix      = GLFW_KEY_KP_6,        /**!< Numpad six      */
        NumSeven    = GLFW_KEY_KP_7,        /**!< Numpad seven    */
        NumEight    = GLFW_KEY_KP_8,        /**!< Numpad eight    */
        NumNine     = GLFW_KEY_KP_9,        /**!< Numpad nine     */
        NumDecimal  = GLFW_KEY_KP_DECIMAL,  /**!< Numpad decimal  */
        NumDivide   = GLFW_KEY_KP_DIVIDE,   /**!< Numpad divide   */
        NumMultiply = GLFW_KEY_KP_MULTIPLY, /**!< Numpad multiply */
        NumSubtract = GLFW_KEY_KP_SUBTRACT, /**!< Numpad subtract */
        NumAdd      = GLFW_KEY_KP_ADD,      /**!< Numpad add      */
        NumEnter    = GLFW_KEY_KP_ENTER,    /**!< Numpad enter    */
        NumEqual    = GLFW_KEY_KP_EQUAL     /**!< Numpad equal    */
    };

    /** @brief */
    enum class Modifier : uint8_t
    {
        Shift = GLFW_MOD_SHIFT,
        Ctrl  = GLFW_MOD_CONTROL,
        Alt   = GLFW_MOD_ALT,
        Super = GLFW_MOD_SUPER
    };

    /** @brief */
    typedef EnumSet<Modifier, 0xff> ModifierFlags;

    /** @brief */
    enum class EventType : size_t
    {
        UNKNOWN,
        MOUSE_MOVE,
        MOUSE_BUTTON_PRESSED,
        MOUSE_BUTTON_RELEASED,
        MOUSE_SCROLL,
        KEY_PRESSED,
        KEY_RELEASED,
        KEY_REPEAT,
        TEXT_INPUT,
        VIEWPORT_CONFIGURATION_CHANGED,
    };

    /** @brief */
    struct UserEvent
    {
        EventType     Type;
        math::ivec2   MousePosition;
        math::vec2    MouseDelta;
        Key           KeyCode;
        ModifierFlags Modifiers;
    };

    // /** @brief */
    // struct LayoutSpecification
    // {
    //     math::ivec2 WindowSize;
    //     math::vec2  DpiScaling;
    //     math::ivec2 FramebufferSize;
    // };

    class Engine;

    /** @class Window
     *
     *  Wrapper around a GLFW3 window.
     */
    class IWindow
    {
      public:
        /** @brief Constructor
         *
         * Constructs a window with the given size and title.
         *
         * @param aWidth  The desired width of the window
         * @param aHeight The desired height of the window
         * @param aTitle  Window title
         */
        IWindow( int aWidth, int aHeight, std::string aTitle );

        /** @brief Destructor */
        ~IWindow();

        /** @brief Not copyable */
        IWindow( const IWindow & )            = delete;
        IWindow &operator=( const IWindow & ) = delete;

        /** @brief Should we close the window
         *
         * @return `true` is the user input an event which requires the current window to close,
         *         such as clicking on the 'close' button.
         */
        bool WindowShouldClose() { return glfwWindowShouldClose( mWindow ); }

        /** @brief Poll events from the underlying window system
         *
         *  All registered callbacks for this window will be triggered for the appropriate events.
         */
        void PollEvents() { glfwPollEvents(); }

        /** @brief Retrieves the size of the window in a format suitable for Vulkan functions
         *
         * @returns VkExtent2D instance holding the window size.
         */
        VkExtent2D GetExtent() { return { static_cast<uint32_t>( mWidth ), static_cast<uint32_t>( mHeight ) }; }

        /** @brief Check to see if the window was resized during the last frame
         *
         * When the window is resized all resources should be recreated to account for the new size.
         *
         * @returns `true` if the window was resized.
         */
        bool WindowWasResized() { return mFramebufferWasResized; }

        /** @brief Reset the resized flag */
        void ResetWindowResizedFlag() { mFramebufferWasResized = false; }

        /** @brief Create a Vulkan surface for this window.
         *
         * Throws a std::runtime_error if the surface could not be created.
         *
         * @param  aInstance The Vulkan instance that the surface wil be created in
         * @param  o_Surface  Pointer to the location the surface is stored in.
         */
        void CreateWindowSurface( VkInstance aInstance, VkSurfaceKHR *o_Surface );

        /** @brief */
        math::vec2 GetMainWindowSize();

        /** @brief */
        math::ivec2 GetFramebufferSize();

        /** @brief */
        GLFWwindow *GetGLFWWindow() { return mWindow; }

        /** @brief */
        void SetEngineLoop( Engine *aEngineLoop ) { mEngineLoop = aEngineLoop; }

        /** @brief */
        void SetTitle( std::string aTitle ) { glfwSetWindowTitle( mWindow, aTitle.c_str() ); }

        math::ivec2 GetMousePosition();

        static void InitializeWindowingBackend();
        static void ShutdownWindowingBackend();

      private:
        static void OnFramebufferResize( GLFWwindow *window, int width, int height );
        static void OnWindowClose( GLFWwindow *window );
        static void OnWindowRefresh( GLFWwindow *window );
        static void OnKey( GLFWwindow *window, const int key, int, const int action, const int mods );
        static void OnMouseButton( GLFWwindow *window, const int button, const int action, const int mods );
        static void OnCursorPosition( GLFWwindow *window, const double x, const double y );
        static void OnMouseScroll( GLFWwindow *window, const double x, const double y );
        static void OnTextInput( GLFWwindow *window, unsigned int codepoint );
        static void OnGLFWError( int error, const char *description );

        void InitializeWindow();

        int  mWidth;
        int  mHeight;
        bool mFramebufferWasResized = false;

        math::ivec2 mLastMousePosition;

        std::string mWindowName = "";
        GLFWwindow *mWindow     = nullptr;
        Engine     *mEngineLoop = nullptr;
    };
} // namespace SE::Core
