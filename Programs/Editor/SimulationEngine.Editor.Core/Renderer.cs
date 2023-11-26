using System.Runtime.InteropServices;

namespace SimulationEngine.Editor.Core;

public class Renderer
{
    IntPtr _sceneRenderer = 0;

    public Renderer(int sampleCount, int colorFormat)
    {
        _sceneRenderer = CreateRenderer(Engine.GraphicContext, sampleCount, colorFormat);
    }

    [DllImport(Config.EnginePath)]
    private static extern IntPtr CreateRenderer(IntPtr context, int sampleCount, int colorFormat);
}
