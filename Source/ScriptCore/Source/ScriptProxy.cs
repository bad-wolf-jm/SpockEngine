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

    // public class RemoteScriptProxy : MarshalByRefObject, IScriptProxy
    // {
    //     object mScriptObject;

    //     public RemoteScriptProxy(object aScriptObject)
    //     {
    //         mScriptObject = aScriptObject;
    //     }

    //     public void Initialize(StreamWriter aConsoleOut)
    //     {
    //         if (mScriptObject == null) return;

    //         var lInitMethod = mScriptObject.GetType().GetMethod("Initialize");
    //         lInitMethod.Invoke(mScriptObject, new object[] { aConsoleOut });
    //     }

    //     public void Shutdown()
    //     {
    //         if (mScriptObject == null) return;

    //         var lInitMethod = mScriptObject.GetType().GetMethod("Shutdown");
    //         lInitMethod.Invoke(mScriptObject, null);
    //     }

    //     public string[] GetScriptNames()
    //     {
    //         if (mScriptObject == null) return new string[0];

    //         var lGetScriptNamesMethod = mScriptObject.GetType().GetMethod("GetScriptNames");
    //         return lGetScriptNamesMethod.Invoke(mScriptObject, null) as string[];
    //     }

    //     public IScript Instantiate(string aName)
    //     {
    //         if (mScriptObject == null) return new RemoteScript(null);

    //         var lInstantiateMethod = mScriptObject.GetType().GetMethod("Instantiate");
    //         return new RemoteScript(lInstantiateMethod.Invoke(mScriptObject, new object[] { aName }));
    //     }
    // }

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