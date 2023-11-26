using System.Runtime.InteropServices;

namespace SimulationEngine.Editor.Core;

public class Engine
{
    public static readonly IntPtr GraphicContext = 0;

    static Engine()
    {
        GraphicContext = CreateGraphicContext(4);
    }

    [DllImport("LTSimulationEngine.dll")]
    public static extern IntPtr CreateGraphicContext(int SampleCount);
}
