using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

using SpockEngine;
using SpockEngine.Math;

using Metrino.Otdr;
using Metrino.Otdr.Instrument;
using Metrino.Otdr.SignalProcessing;

using Metrino.Interop;

public class ScriptProxy : MarshalByRefObject, IScriptProxy
{
    Dictionary<string, Type> mScripts;

    public void Initialize(StreamWriter aConsoleOut)
    {
        mScripts = new Dictionary<string, Type>();
        
        if (aConsoleOut != null)
            Console.SetOut(aConsoleOut);

        Console.WriteLine($"Hello 'World'");
        foreach(var x in Utilities.GetAllDerivedTypes<IScript>())
            mScripts[x.FullName] = x;
    }

    public void Shutdown()
    {
        Console.WriteLine($"Good bye 'World'");
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
namespace Test
{
    public class TestScript : MarshalByRefObject, IScript
    {
        public TestScript() { }

        public void Begin()
        {
            System.Console.WriteLine($"BEGIN {DateTime.UtcNow}");
        }

        public void End()
        {
            System.Console.WriteLine($"END {DateTime.UtcNow}");
        }

        public bool Tick(float aTs)
        {
            System.Console.WriteLine($"{DateTime.UtcNow}");

            return true;
        }
    }

    public class TestSorValues : MarshalByRefObject, IScript
    {
        public TestSorValues() { }

        OlmFile mFile;
        SinglePulseTraceCollection mReportingTraces;

        public void Begin()
        {
            var lPath = @"C:\GitLab\SpockEngine\Programs\TestOtdrProject\Resources\TraceFiles\OTDRXPE-835\DUB16-6-1-A-DUB16-L0-A-02-F147.iolm";
            System.Console.WriteLine($"Loading 'DUB16-6-1-A-DUB16-L0-A-02-F147.iolm'...");
            mFile = FileLoader.LoadOlmData(lPath);
            mReportingTraces = mFile.Measurement.ReportingTraces;

            foreach(var lTrace in mReportingTraces)
            {
                lTrace.EventTable.Clear();
                mFile.Measurement.UpdateOtdrEvents(lTrace, true);
                Console.WriteLine($"Wavelength={lTrace.Wavelength} -- Loss={lTrace.SpansLoss}, ORL={lTrace.SpansOrl}");
                System.Console.WriteLine($"{lTrace.Wavelength}");
            }

        }

        public void End()
        {
        }

        public bool Tick(float aTs)
        {
            return false;
        }
    }

}