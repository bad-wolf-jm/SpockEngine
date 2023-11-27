using CommunityToolkit.Mvvm.ComponentModel;
using SimulationEngine.Editor.Core;

namespace SimulationEngine.Editor.ViewModels
{
    public partial class MainWindowViewModel : ObservableObject
    {
        Renderer _renderer;
        Scene _scene;

        public string Greeting => "Welcome to Avalonia!";

        public MainWindowViewModel()
        {
            _renderer = new Renderer(4, 12);
            _scene = new Scene();
        }
    }
}