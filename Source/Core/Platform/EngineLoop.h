/** @file */
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "UI/UI.h"

#include "entt/entt.hpp"

#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//SwapChain.h"
#include "Core/Optix/OptixContext.h"
#include "Window.h"

#include "Core/GraphicContext//UI/UIContext.h"

/** @brief */
namespace LTSE::Core
{

    /** @class EngineLoop
     *
     * This class encapsulates the main engine loop. It is responsible for all the initialization
     * required to run a 3D application. This class is designed to be used either in a main function
     * which handles the game loop, or controlled by an external tick function.
     *
     */
    class EngineLoop
    {
      public:
        friend class Window;
        EngineLoop()  = default;
        ~EngineLoop() = default;

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
        void Shutdown();

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

        ///** @brief Retrieve the underlying ImGui context */
        // ImGUIOverlay* GetImGUIOverlay() { return s_ImGUIOverlay; }

        /** @brief Retrieve the font used for icons. */
        ImFont *GetIconFont() { return s_IconFont; }

        /** @brief Retrieve the font used for normal text. */
        ImFont *GetNormalFont() { return s_MainFont; }

        /** @brief Retrieve the font used for bold text. */
        ImFont *GetBoldFont() { return s_MainFont; }

        math::ivec2 GetViewportSize() { return m_ViewportClient->GetFramebufferSize(); }

        /** @brief REtrieve the ImGui IO context. */
        // ImGuiIO& GetImGuiIO()
        //{
        //     ImGui::SetCurrentContext(GetImGUIOverlay());
        //     return ImGui::GetIO();
        //  }
        LTSE::Graphics::GraphicContext &GetGraphicContext() { return mGraphicContext; }

        LTSE::Graphics::GraphicContext &GetDevice() { return mGraphicContext; }

        Ref<LTSE::Graphics::SwapChainRenderTarget> GetSwapchainRenderer() { return m_SwapChainRenderer; }
        LTSE::Graphics::RenderContext             &GetRenderContext() { return m_RenderContext; }

        Ref<LTSE::Core::UIContext> UIContext() { return mImGUIOverlay; };

        /** @brief Returns the configuration folder */
        const std::string &GetConfigFolder() { return m_LocalConfigFolder; }

        /** @brief Returns the configuration folder */
        const std::string &GetUserHomeFolder() { return m_UserHomeFolder; }

        void SetApplicationName( std::string a_Name ) { m_ApplicationName = a_Name; }

        std::string GetApplicationName() { return m_ApplicationName; }

        void SetInitialWindowPosition( math::ivec2 a_Position );
        void SetInitialWindowSize( math::ivec2 a_Size );
        void SetImGuiConfigurationFile( std::string a_Path );

        math::ivec2 GetWindowPosition();
        math::ivec2 GetWindowSize();
        std::string GetImGuiConfigurationFile();

      private:
        void IOEvent( UserEvent &a_Event );

      private:
        static EngineLoop *s_UniqueInstance;

        LTSE::Graphics::GraphicContext mGraphicContext;

        Ref<LTSE::Core::Window>        m_ViewportClient;
        LTSE::Graphics::GraphicContext m_GraphicContextData{};

        Ref<LTSE::Core::UIContext> mImGUIOverlay;

        Ref<LTSE::Graphics::SwapChain> m_SwapChain;
        LTSE::Graphics::ARenderContext m_SwapChainRenderContext;

        double m_EngineLoopStartTime;
        double m_LastFrameTime = 0.0f;

        math::ivec2 m_InitialMainWindowSize     = { 1920, 1080 };
        math::ivec2 m_InitialMainWindowPosition = { 100, 100 };
        std::string mImGuiConfigPath            = "imgui.ini";

        math::ivec2 m_MainWindowSize;
        math::vec2  m_DpiScaling;
        math::ivec2 m_FramebufferSize;
        math::ivec2 s_LastMousePosition;

        ImFont *s_MainFont;
        ImFont *s_IconFont;
        UIStyle s_UIStyle;

        std::string m_ApplicationName   = "";
        std::string m_LocalConfigFolder = "";
        std::string m_UserHomeFolder    = "";
    };

} // namespace LTSE::Core