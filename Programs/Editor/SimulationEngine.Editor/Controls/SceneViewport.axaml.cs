using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Media.Imaging;
using Avalonia.Rendering;
using SimulationEngine.Editor.Core;
using System;
using System.Drawing;
using System.Reflection.Emit;

namespace SimulationEngine.Editor.Desktop.Controls;

public class SceneViewport : TemplatedControl
{
    Renderer _renderer;

    public static readonly StyledProperty<Renderer> RendererProperty =
        AvaloniaProperty.Register<SceneViewport, Renderer>(nameof(Renderer), defaultValue: null);

    public Renderer Renderer
    {
        get => GetValue(RendererProperty);
        set => SetValue(RendererProperty, value);
    }

    public static readonly StyledProperty<Scene> SceneProperty =
        AvaloniaProperty.Register<SceneViewport, Scene>(nameof(Scene), defaultValue: null);

    public Scene Scene
    {
        get => GetValue(SceneProperty);
        set => SetValue(SceneProperty, value);
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
        _renderer = new Renderer(4, 12);

        SizeChanged += SceneViewport_SizeChanged;

        FramebufferProperty.Changed.AddClassHandler<SceneViewport>(OnFramebufferChanged);
        RendererProperty.Changed.AddClassHandler<SceneViewport>(OnRendererChanged);
        SceneProperty.Changed.AddClassHandler<SceneViewport>(OnSceneChanged);
    }


    Avalonia.Controls.Image? _image;
    public void OnApplyTemplate(TemplateAppliedEventArgs e)
    {
        _image = e.NameScope.Find<Avalonia.Controls.Image>("image");
    }

    private void OnSceneChanged(SceneViewport viewport, AvaloniaPropertyChangedEventArgs args)
    {
        UpdateFramebuffer();
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
        _renderer.ResizeOutput((int)e.NewSize.Width, (int)e.NewSize.Height);

        var size = new PixelSize(_renderer.OutputWidth, _renderer.OutputHeight);
        var dpi = new Vector(96, 96);
        Framebuffer = new WriteableBitmap(size, dpi, Avalonia.Platform.PixelFormats.Rgba8888);
    }

    private void UpdateFramebuffer()
    {
        if (Scene == null) return;
        if (Framebuffer == null) return;

        using (var bitmapData = Framebuffer.Lock())
        {
            _renderer.Update(Scene);
            _renderer.Render(bitmapData.Address);
        }

        if (_image != null)
            _image.InvalidateVisual();
    }
}