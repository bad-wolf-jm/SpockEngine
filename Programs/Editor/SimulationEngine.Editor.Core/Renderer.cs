using System.Runtime.InteropServices;

namespace SimulationEngine.Editor.Core;

public class Renderer
{
    IntPtr _sceneRenderer = 0;

    public Renderer(int sampleCount, int colorFormat)
    {
        _sceneRenderer = CreateRenderer(Engine.GraphicContext, sampleCount, colorFormat);
    }

    public void Update(Scene scene)
    {
        UpdateRenderer(_sceneRenderer, scene.Handle);
    }

    [DllImport(Config.EnginePath)]
    private static extern IntPtr CreateRenderer(IntPtr context, int sampleCount, int colorFormat);

    [DllImport(Config.EnginePath)]
    private static extern IntPtr UpdateRenderer(IntPtr self, IntPtr scene);
}
