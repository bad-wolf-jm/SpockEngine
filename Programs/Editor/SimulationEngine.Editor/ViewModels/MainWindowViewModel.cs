using CommunityToolkit.Mvvm.ComponentModel;
using SimulationEngine.Editor.Core;

namespace SimulationEngine.Editor.ViewModels
{
    public partial class MainWindowViewModel : ObservableObject
    {
        //Scene _scene;
        Renderer _renderer;

        public string Greeting => "Welcome to Avalonia!";

        public MainWindowViewModel()
        {
            _renderer = new Renderer(4, 12);
        }
    }
}