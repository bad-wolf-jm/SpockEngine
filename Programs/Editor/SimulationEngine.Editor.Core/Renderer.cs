using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Media.Imaging;

namespace SimulationEngine.Editor.Core;

public class Renderer
{
    IntPtr _sceneRenderer = 0;
    WriteableBitmap _bitmap;
    public WriteableBitmap Output => _bitmap;

    int _outputWidth = 0;
    int _outputHeight = 0;
    int _bitsPerPixel = 0;

    public Renderer(int sampleCount, int colorFormat)
    {
        _sceneRenderer = CreateRenderer(Engine.GraphicContext, sampleCount, colorFormat);
        _bitsPerPixel = (int)GetPixelSize(_sceneRenderer);
    }

    public void Update(Scene scene)
    {
        UpdateRenderer(_sceneRenderer, scene.Handle);
    }

    public void Render()
    {
        Render(_sceneRenderer);
        UpdateRenderedImage();
    }

    public void ResizeOutput(int width, int height)
    {
        var size = new PixelSize(width, height);
        var dpi = new Vector(96, 96);
        _bitmap = new WriteableBitmap(size, dpi, Avalonia.Platform.PixelFormats.Rgba64);

        ResizeRendererOutput(_sceneRenderer, width, height);
    }

    public void UpdateRenderedImage()
    {
        uint width, height;
        byte bpp;
        GetOutputSize(_sceneRenderer, out width, out height, out bpp);

        using (var bitmapData = _bitmap.Lock())
        {
            GetRenderedImage(_sceneRenderer, bitmapData.Address, (int)(width * height * bpp));
        }
        _bitmap.Save(@"D:\Personal\Git\SpockEngine\Programs\Editor\test_2.png");
    }

    [DllImport(Config.EnginePath)]
    private static extern IntPtr CreateRenderer(IntPtr context, int sampleCount, int colorFormat);

    [DllImport(Config.EnginePath)]
    private static extern IntPtr UpdateRenderer(IntPtr self, IntPtr scene);

    [DllImport(Config.EnginePath)]
    private static extern IntPtr Render(IntPtr self);

    [DllImport(Config.EnginePath)]
    private static extern IntPtr ResizeRendererOutput(IntPtr self, int width, int height);

    [DllImport(Config.EnginePath)]
    private static extern void GetRenderedImage(IntPtr self, IntPtr buffer, int len);

    [DllImport(Config.EnginePath)]
    private static extern byte GetPixelSize(IntPtr context);

    [DllImport(Config.EnginePath)]
    private static extern IntPtr GetOutputSize(IntPtr self, out uint width, out uint height, out byte bpp);
}
