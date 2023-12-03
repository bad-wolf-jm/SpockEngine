using Avalonia;
using Avalonia.Controls.Primitives;
using Avalonia.Media.Imaging;
using Avalonia.Rendering;
using SimulationEngine.Editor.Core;
using System;
using System.Reflection.Emit;

namespace SimulationEngine.Editor.Desktop.Controls;

public class SceneViewport : TemplatedControl
{
    public static readonly StyledProperty<Renderer> RendererProperty =
        AvaloniaProperty.Register<SceneViewport, Renderer>(nameof(Renderer), defaultValue: null);

    public Renderer Renderer
    {
        get => GetValue(RendererProperty);
        set => SetValue(RendererProperty, value);
    }

    public static readonly StyledProperty<WriteableBitmap> FramebufferProperty =
        AvaloniaProperty.Register<SceneViewport, WriteableBitmap>(nameof(Framebuffer), defaultValue: null);

    public WriteableBitmap Framebuffer
    {
        get => GetValue(FramebufferProperty);
        set => SetValue(FramebufferProperty, value);
    }

    public SceneViewport()
    {
        SizeChanged += SceneViewport_SizeChanged;

        FramebufferProperty.Changed.AddClassHandler<SceneViewport>(OnFramebufferChanged);
        RendererProperty.Changed.AddClassHandler<SceneViewport>(OnRendererChanged);

    }

    private void OnFramebufferChanged(SceneViewport viewport, AvaloniaPropertyChangedEventArgs args)
    {
        UpdateFramebuffer();
    }

    private void OnRendererChanged(SceneViewport viewport, AvaloniaPropertyChangedEventArgs args)
    {
        UpdateFramebuffer();
    }

    private void SceneViewport_SizeChanged(object? sender, Avalonia.Controls.SizeChangedEventArgs e)
    {
        if (Renderer == null) return;

        var newSize = e.NewSize;
        Renderer.ResizeOutput((int)newSize.Width, (int)newSize.Height);

        var size = new PixelSize(Renderer.OutputWidth, Renderer.OutputHeight);
        var dpi = new Vector(96, 96);
        Framebuffer = new WriteableBitmap(size, dpi, Avalonia.Platform.PixelFormats.Rgba8888);
    }

    private void UpdateFramebuffer()
    {
        if (Renderer == null) return;
        if (Framebuffer == null) return;

        using (var bitmapData = Framebuffer.Lock())
            Renderer.Render(bitmapData.Address);
    }
}