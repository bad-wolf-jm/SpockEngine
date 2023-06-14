using System;
using System.Reflection;
using System.Runtime.Loader;

namespace SpockEngine
{
    public class ScriptLoadContext : AssemblyLoadContext
    {
        private AssemblyDependencyResolver mResolver;

        public ScriptLoadContext(string pluginPath)
        {
            mResolver = new AssemblyDependencyResolver(pluginPath);
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
    }
}