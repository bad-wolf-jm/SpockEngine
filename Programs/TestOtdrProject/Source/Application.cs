using System;
using System.Linq;
using System.IO;

using SpockEngine;
using SpockEngine.Math;

using Metrino.Otdr;
using Metrino.Otdr.Instrument;
using Metrino.Otdr.SignalProcessing;

using Metrino.Interop;

public class ScriptProxy : MarshalByRefObject, IScriptProxy
{
    public void Initialize(StreamWriter aConsoleOut)
    {
        if (aConsoleOut != null)
            Console.SetOut(aConsoleOut);

        Console.WriteLine($"Hello 'World'");
    }

    public void Shutdown()
    {
        Console.WriteLine($"Good bye 'World'");
    }

    public string[] GetScriptNames()
    {
        return new string[] { "Test.TestSorValues" };
    }

    public IScript Instantiate(string aName)
    {
        return (new Test.TestSorValues()) as IScript;
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