
#include "Engine/Engine.h"
#include "Core/GraphicContext//UI/UIContext.h"

#include "UI/Widgets.h"

#include <fmt/core.h>

#include "Core/Logging.h"

#include "UI/CanvasView.h"
#include "UI/UI.h"
#include "UI/Widgets.h"

#include "blueprints-example.h"s

using namespace LTSE::Core;
using namespace LTSE::Graphics;
using namespace LTSE::Core::UI;
using namespace math::literals;

static EngineLoop g_EngineLoop;
static std::shared_ptr<Example> g_example;

void Update( Timestep ts ) {}

bool RenderUI( ImGuiIO &io )
{
    auto l_WindowSize = UI::GetRootWindowSize();
    bool Quit         = false;

    static bool p_open = true;
    ImGui::Begin("NODES", &p_open, ImGuiWindowFlags_None);
    g_example->OnFrame( 0.0f );
    ImGui::End();

    return Quit;
}

int main( int argc, char **argv )
{
    g_EngineLoop = EngineLoop();
    g_EngineLoop.PreInit( 0, nullptr );
    g_EngineLoop.Init();

    g_EngineLoop.UIDelegate.connect<RenderUI>();
    g_EngineLoop.UpdateDelegate.connect<Update>();

    g_example = std::make_shared<Example>();
    g_example->OnStart();

    while( g_EngineLoop.Tick() )
    {
    };

    return 0;
}
