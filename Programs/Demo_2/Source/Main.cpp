
#include "Engine/Engine.h"
#include "Core/GraphicContext//UI/UIContext.h"

#include "UI/Widgets.h"

#include <fmt/core.h>

#include "Core/Logging.h"

#include "UI/CanvasView.h"
#include "UI/UI.h"
#include "UI/Widgets.h"

#include "Editor/BaseEditorApplication.h"
#include "Editor/EditorWindow.h"

using namespace SE::Core;
using namespace SE::Graphics;
using namespace SE::Editor;
using namespace SE::Core::UI;
using namespace math::literals;

static EngineLoop g_EngineLoop;
static BaseEditorApplication g_EditorWindow;

int main( int argc, char **argv )
{
    g_EditorWindow.ApplicationName = "TEST";
    g_EditorWindow.WindowSize      = { 2920, 1580 };
    g_EditorWindow.Init();

    return g_EditorWindow.Run();
}
