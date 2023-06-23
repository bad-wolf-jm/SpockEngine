using System;
using System.IO;
using System.Linq;

using System.Collections.Generic;

namespace SpockEngine
{
    public interface IScriptProxy
    {
        void Initialize(StreamWriter aConsoleOut);

        void Shutdown();

        string[] GetScriptNames();

        IScript Instantiate(string aName);
    }

    public class ScriptProxy : MarshalByRefObject, IScriptProxy
    {
        Dictionary<string, Type> mScripts;

        public void Initialize(StreamWriter aConsoleOut)
        {
            mScripts = new Dictionary<string, Type>();

            if (aConsoleOut != null)
                Console.SetOut(aConsoleOut);

            foreach (var x in Utilities.GetAllDerivedTypes<IScript>().Where(x => (x.FullName != "SpockEngine.Script")))
                mScripts[x.FullName] = x;
        }

        public void Shutdown()
        {
        }

        public string[] GetScriptNames()
        {
            return mScripts.Keys.ToArray();
        }

        public IScript Instantiate(string aName)
        {
            return Activator.CreateInstance(mScripts[aName]) as IScript;
        }
    }
}