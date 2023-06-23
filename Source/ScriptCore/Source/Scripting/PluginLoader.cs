using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Remoting.Lifetime;
using System.Security.Permissions;

namespace SpockEngine.Scripting
{
    /// <summary>
    /// An instance of this class is created in the plugin's AppDomain.  It's
    /// responsible for loading the plugin assembly, and has calls that
    /// allow the host to query and remotely instantiate objects.
    /// <para>
    /// Because the object "lives" in the plugin AppDomain but is called
    /// from the host AppDomain, it must derive from MarshalByRefObject.
    /// </para>
    /// </summary>
    public sealed class PluginLoader : MarshalByRefObject
    {
        public PluginLoader()
        {
            Console.WriteLine("Hello from the other side, id=" + AppDomain.CurrentDomain.Id);
        }

        public void SetConsoleOut(StreamWriter aConsoleOut)
        {
            if (aConsoleOut != null)
                Console.SetOut(aConsoleOut);
        }

        /// <summary>
        /// Loads the assembly in the specified DLL, finds the first
        /// concrete class that implements IPlugin, and instantiates it.
        /// </summary>
        /// <param name="dllPath">Absolute path to DLL.</param>
        public IPlugin Load(string dllPath)
        {
            Assembly asm = Assembly.LoadFile(dllPath);

            foreach (Type type in asm.GetTypes())
            {
                if (type.IsClass && !type.IsAbstract && type.GetInterfaces().Contains(typeof(IPlugin)))
                {
                    ConstructorInfo ctor = type.GetConstructor(Type.EmptyTypes);
                    IPlugin plugin = (IPlugin)ctor.Invoke(null);
                    Console.WriteLine("Created instance: " + plugin);

                    return plugin;
                }
            }

            throw new Exception("No IPlugin class found");
        }

        /// <summary>
        /// Allows the host to ping the loader class in the plugin AppDomain.
        /// Strictly for debugging.
        /// </summary>
        public int Ping(int val)
        {
            Console.WriteLine("PluginLoader(id=" + AppDomain.CurrentDomain.Id + "): ping " + val);

            return val + 1;
        }
    }
}
