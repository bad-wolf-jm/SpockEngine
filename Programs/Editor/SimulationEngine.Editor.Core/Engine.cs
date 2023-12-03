using System.Runtime.InteropServices;

namespace SimulationEngine.Editor.Core;

public class Engine
{
    public static readonly IntPtr GraphicContext = 0;

    static Engine()
    {
        GraphicContext = CreateGraphicContext(4);
        SetShaderCacheFolder(@"C:\Users\jmalb\AppData\Local\SpockEngine\Resources\Shaders");
    }

    public static void Shutdown()
    {
    }

    [DllImport(Config.EnginePath)]
    public static extern IntPtr CreateGraphicContext(int SampleCount);

    [DllImport(Config.EnginePath)]
    public static extern void SetShaderCacheFolder(string SampleCount);
}
