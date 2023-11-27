using System.Runtime.InteropServices;

namespace SimulationEngine.Editor.Core;

public class Scene
{
    IntPtr _scene = 0;

    public Scene()
    {
        _scene = CreateScene(Engine.GraphicContext);
    }

    public IntPtr Handle { get => _scene; }

    public void Update(float ts)
    {
        UpdateScene(_scene, ts);
    }

    public void LoadScenario(string path)
    {
        LoadScenario(_scene, path);
    }

    [DllImport(Config.EnginePath)]
    private static extern IntPtr CreateScene(IntPtr context);

    [DllImport(Config.EnginePath)]
    private static extern void UpdateScene(IntPtr self, float ts);

    [DllImport(Config.EnginePath)]
    private static extern void LoadScenario(IntPtr self, [MarshalAs(UnmanagedType.LPStr)] string ts);
}
