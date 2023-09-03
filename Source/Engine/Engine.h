/** @file */
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <mutex>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "UI/UI.h"
#include "UI/UIContext.h"

#include "entt/entt.hpp"

// #include "Core/Optix/OptixContext.h"
#include "Graphics/Interface/IWindow.h"
#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkRenderContext.h"
#include "Graphics/Vulkan/VkSwapChain.h"
#include "Graphics/Vulkan/VkSwapChainRenderContext.h"
// #include "Core/Optix/OptixContext.h"
/** @brief */
namespace SE::Core
{

    /** @class Engine
     *
     * This class encapsulates the main engine loop. It is responsible for all the initialization
     * required to run a 3D application. This class is designed to be used either in a main function
     * which handles the game loop, or controlled by an external tick function.
     *
     */
    class Engine
    {
      public:
        friend class IWindow;

        Engine()  = default;
        ~Engine() = default;

        entt::delegate<bool( ImGuiIO & )>   UIDelegate{};
        entt::delegate<void( Timestep )>    UpdateDelegate{};
        entt::delegate<void( void )>        RenderDelegate{};
        entt::delegate<void( UserEvent & )> IOEventDelegate{};

        /** @brief Pre-initialization
         *
         * Performs the necessary pre-initialization steps. This includes reading and parsing the command line,
         * creating the configuration folder if required, loading any existing configuration, and creating the
         * main window and user interface contexts.
         */
        void PreInit( int argc, char **argv );

        /** @brief Initialization
         *
         * Everything that needs to be initialized after the preinitialization phase.
         */
        void Init();

        /** @brief Shutdown
         *
         * Releases all resources which were acquired by the engine loop, and shuts down Live++ if required.
         */
        // void Shutdown();

        /** @brief Tick function
         *
         * This is the main tick function. It sets up a new user interface frame, and hands over control to the
         * `UIDelegate`, the `UpdateDelegate` and the `RenderDelegate` in that order.
         * - `UIDelegate` should render any UI components
         * - `UpdateDelegate` should update the 3D world and the physics simulations if any
         * - `RenderDelegate` performs the actual rendering.
         *
         * At the end of this funciton, the user interface which was set up during the call to `UIDelegate` is
         * rendered on top of the scene. The function then polls the underlying windowing system for events which
         * are handed to the `IOEventDelegate`.
         *
         * @returns `true` if the engine loop should exit, and `false` to keep going.
         */
        bool Tick();

        /** @brief Current time in milliseconds since the epoch. */
        int64_t GetTime();

        /** @brief Time elapsed since the class was created.*/
        int64_t GetElapsedTime();

        /** @brief Retrieved the underlying application window. */
        GLFWwindow *GetMainApplicationWindow();

        math::ivec2 GetViewportSize()
        {
            return mViewportClient->GetFramebufferSize();
        }

        SE::Graphics::ref_t<IGraphicContext> GetGraphicContext()
        {
            return mGraphicContext;
        }
        ref_t<SE::Core::UIContext> UIContext()
        {
            return mImGUIOverlay;
        };

        void SetApplicationName( string_t a_Name )
        {
            mApplicationName = a_Name;
        }
        string_t GetApplicationName()
        {
            return mApplicationName;
        }

        void SetInitialWindowPosition( math::ivec2 a_Position );
        void SetInitialWindowSize( math::ivec2 a_Size );
        void SetImGuiConfigurationFile( string_t a_Path );

        math::ivec2 GetWindowPosition();
        math::ivec2 GetWindowSize();
        string_t GetImGuiConfigurationFile();

        static void Initialize( math::ivec2 aInitialMainWindowSize, math::ivec2 aInitialMainWindowPosition, fs::path aImGuiConfigPath,
                                UIConfiguration const &aUIConfiguration );
        static void Shutdown();

        static std::unique_ptr<Engine> &GetInstance()
        {
            return mUniqueInstance;
        };

        void SubmitToMainThread( const std::function<void()> &aThunk );
        void ExecuteMainThreadQueue();

      private:
        void IOEvent( UserEvent &a_Event );

      private:
        static std::unique_ptr<Engine> mUniqueInstance;

        std::vector<std::function<void()>> mMainThreadQueue;
        std::mutex                         mMainThreadQueueMutex;

        ref_t<SE::Core::IWindow>             mViewportClient;
        ref_t<SE::Graphics::IGraphicContext> mGraphicContext = nullptr;

        ref_t<SE::Core::UIContext> mImGUIOverlay;

        ref_t<SE::Graphics::ISwapChain>     mSwapChain;
        ref_t<SE::Graphics::IRenderContext> mSwapChainRenderContext;

        double mEngineLoopStartTime;
        double mLastFrameTime = 0.0f;

        math::ivec2     mInitialMainWindowSize     = { 1920, 1080 };
        math::ivec2     mInitialMainWindowPosition = { 100, 100 };
        string_t     mImGuiConfigPath           = "imgui.ini";
        UIConfiguration mUIConfiguration{};

        math::ivec2 mMainWindowSize;
        math::vec2  mDpiScaling;
        math::ivec2 mFramebufferSize;
        math::ivec2 mLastMousePosition;

        string_t mApplicationName = "";
    };

} // namespace SE::Core