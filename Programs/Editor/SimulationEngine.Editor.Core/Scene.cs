using System.Runtime.InteropServices;

namespace SimulationEngine.Editor.Core;

public class Scene
{
    IntPtr _scene = 0;

    public Scene()
    {
        _scene = CreateScene(Engine.GraphicContext);
    }

    [DllImport(Config.EnginePath)]
    private static extern IntPtr CreateScene(IntPtr context);
}
