using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using SimulationEngine.Editor.Core;


namespace SimulationEngine.Editor.ViewModels
{
    public partial class MainWindowViewModel : ObservableObject
    {
        Renderer _renderer;
        public Renderer Renderer => _renderer;
        Scene _scene;

        public string Greeting => "Welcome to Avalonia!";

        public Bitmap Viewport => _renderer.Output;

        public MainWindowViewModel()
        {
            // set shader cache path

            _renderer = new Renderer(4, 12);
            _renderer.ResizeOutput(1000, 700);

            _scene = new Scene();

            _scene.LoadScenario("C:\\GitLab\\SpockEngine\\Saved\\TEST\\Sponza_SCENE\\SceneDefinition.yaml");
            _scene.Update(0.0f);

            _renderer.Update(_scene);
            _renderer.Render();

            //var x = _renderer.GetRenderedImage();
        }
    }
}