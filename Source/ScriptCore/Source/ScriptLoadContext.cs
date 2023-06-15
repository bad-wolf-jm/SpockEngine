using System;
using System.IO;
using System.Reflection;
using System.Runtime.Loader;

namespace SpockEngine
{
    // public class ScriptProxy : IScriptProxy
    // {

    // }

    public class ScriptLoadContext : AssemblyLoadContext
    {
        private string mPath;
        private AssemblyDependencyResolver mResolver;
        private Assembly mAssembly;
        private RemoteScriptProxy mProxy;

        public ScriptLoadContext(string pluginPath)
        {
            mPath = pluginPath;
            mResolver = new AssemblyDependencyResolver(mPath);
            mAssembly = LoadFromAssemblyName(new AssemblyName(Path.GetFileNameWithoutExtension(mPath)));

            Type lScriptType = mAssembly.GetType("ScriptAssemblyDefinition");
            var lScriptObject = Activator.CreateInstance(lScriptType);

            mProxy = new RemoteScriptProxy(Activator.CreateInstance(lScriptType));
        }

        protected override Assembly? Load(AssemblyName assemblyName)
        {
            string assemblyPath = mResolver.ResolveAssemblyToPath(assemblyName);
            if (assemblyPath == null) return null;

            return LoadFromAssemblyPath(assemblyPath);
        }

        protected override IntPtr LoadUnmanagedDll(string unmanagedDllName)
        {
            string libraryPath = mResolver.ResolveUnmanagedDllToPath(unmanagedDllName);
            if (libraryPath == null) return IntPtr.Zero;

            return LoadUnmanagedDllFromPath(libraryPath);
        }

        public RemoteScriptProxy GetProxyObject() { return mProxy; }
    }
}