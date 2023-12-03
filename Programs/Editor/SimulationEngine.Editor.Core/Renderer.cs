using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Media.Imaging;

namespace SimulationEngine.Editor.Core;

public class Renderer
{
    IntPtr _sceneRenderer = 0;
    //WriteableBitmap _bitmap;
    //public WriteableBitmap Output => _bitmap;

    int _outputWidth = 0;
    public int OutputWidth => _outputWidth;

    int _outputHeight = 0;
    public int OutputHeight => _outputHeight;

    byte _bitsPerPixel = 0;

    int _bufferByteSize => _outputWidth * _outputHeight * _bitsPerPixel;

    public Renderer(int sampleCount, int colorFormat)
    {
        _sceneRenderer = CreateRenderer(Engine.GraphicContext, sampleCount, colorFormat);
        _bitsPerPixel = GetPixelSize(_sceneRenderer);
    }

    public void Update(Scene scene)
    {
        UpdateRenderer(_sceneRenderer, scene.Handle);
    }

    public void Render()
    {
        InternalRender(_sceneRenderer);
    }

    public void Render(IntPtr framebuffer)
    {
        Render();
        GetRenderedImage(_sceneRenderer, framebuffer, _bufferByteSize);
    }

    public void ResizeOutput(int width, int height)
    {
        ResizeRendererOutput(_sceneRenderer, width, height);

        uint actualWidth, actualHeight;
        byte bpp;
        GetOutputSize(_sceneRenderer, out actualWidth, out actualHeight, out bpp);

        _outputWidth = (int) actualWidth;
        _outputHeight = (int) actualHeight;
    }


    [DllImport(Config.EnginePath)]
    private static extern IntPtr CreateRenderer(IntPtr context, int sampleCount, int colorFormat);

    [DllImport(Config.EnginePath)]
    private static extern IntPtr UpdateRenderer(IntPtr self, IntPtr scene);

    [DllImport(Config.EnginePath)]
    private static extern IntPtr InternalRender(IntPtr self);

    [DllImport(Config.EnginePath)]
    private static extern IntPtr ResizeRendererOutput(IntPtr self, int width, int height);

    [DllImport(Config.EnginePath)]
    private static extern void GetRenderedImage(IntPtr self, IntPtr buffer, int len);

    [DllImport(Config.EnginePath)]
    private static extern byte GetPixelSize(IntPtr context);

    [DllImport(Config.EnginePath)]
    private static extern IntPtr GetOutputSize(IntPtr self, out uint width, out uint height, out byte bpp);
}
