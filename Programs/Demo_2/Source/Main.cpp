
#include "Developer/Platform/EngineLoop.h"
#include "Developer/GraphicContext/UI/UIContext.h"

#include "Developer/UI/Widgets.h"

#include <fmt/core.h>

#include "Core/Logging.h"

#include "Developer/UI/CanvasView.h"
#include "Developer/UI/UI.h"
#include "Developer/UI/Widgets.h"

#include "Developer/Editor/BaseEditorApplication.h"
#include "Developer/Editor/EditorWindow.h"

using namespace LTSE::Core;
using namespace LTSE::Graphics;
using namespace LTSE::Editor;
using namespace LTSE::Core::UI;
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
