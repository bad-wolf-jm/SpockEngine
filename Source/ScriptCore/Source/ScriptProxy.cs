using System;
using System.IO;


namespace SpockEngine
{
    public interface IScriptProxy
    {
        void Initialize(StreamWriter aConsoleOut);
        
        void Shutdown();

        string[] GetScriptNames();

        IScript Instantiate(string aName);
    }
}