// This file is shared between the "host" and "plugin" sides.  It's
// compiled into an assembly that is included by both.

namespace SpockEngine.Scripting
{

    /// <summary>
    /// Methods provided by the plugin.  This code is called from the main
    /// program, and executes in the plugin AppDomain.
    /// 
    /// All arguments and return values must be serializable.
    /// </summary>
    public interface IPlugin
    {
        /// <summary>
        /// Initializes the plugin with an object that can be used to call
        /// methods in the host (to demonstrate two-way communication).
        /// </summary>
        /// <param name="hostObj"></param>
        void SetHostObj(IHost hostObj);

        /// <summary>
        /// Performs a simple test, calling back into the host and returning
        /// the modified result to the host.  If the host object is not
        /// available, returns zero.
        /// </summary>
        int TestRoundTrip(int arg);

        /// <summary>
        /// Compiles the provided script.  This must be done before calls
        /// to TestScriptRoundTrip().  Returns null on error.  The full
        /// compiler error collection object is also returned.
        /// </summary>
        IScript CompileScript(string script, out System.CodeDom.Compiler.CompilerErrorCollection cec);

    }

    /// <summary>
    /// Methods provided by the scripts compiled by the plugin.  This code is
    /// called from the main program, and executes in the plugin AppDomain.
    /// 
    /// All arguments and return values must be serializable.
    /// </summary>
    public interface IScript
    {
        /// <summary>
        /// Similar to TestRoundTrip(), but invokes a method inside the
        /// compiled script.
        /// </summary>
        int TestScriptRoundTrip(IHost hostObj, int arg);
    }

    /// <summary>
    /// Methods provided by the host.  This code is called from the plugin, and
    /// executes in the primary AppDomain.
    /// 
    /// All arguments and return values must be serializable.
    /// </summary>
    public interface IHost
    {
        /// <summary>
        /// Performs a simple test, returning a modified copy of the argument.
        /// </summary>
        int DoSomethingNifty(int arg);
    }

}
