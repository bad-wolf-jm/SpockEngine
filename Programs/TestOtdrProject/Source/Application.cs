using System;
using System.Linq;

using SpockEngine;

using Metrino.Otdr;
using Metrino.Otdr.Instrument;
using Metrino.Otdr.SignalProcessing;

using Metrino.Interop;

namespace Test
{
    public class App : SEApplication
    {
        UIForm mMainForm;

        public App() { }

        public void Initialize() { 
            mMainForm = new UIForm();
            mMainForm.SetTitle("This is a test!!");
        }

        public void Shutdown() { }

        public void Update(float aTs)
        {
            // Console.WriteLine("Update");
            mMainForm.Update();
        }

        public void UpdateUI(float aTs)
        {
            // Console.WriteLine("Update");

        }
    }

    public class TestScript : Script
    {

        private Instrument7000 mSource;
        private Instrument7000 mPowerMeter;

        private bool mSourceStarted;
        private bool mPowerMeterStarted;

        private BlinkDetection mBlinkDetection;


        public TestScript() : base() { }

        override public void BeginScenario()
        {
            base.BeginScenario();
        }

        override public void EndScenario()
        {
            base.EndScenario();
        }

        override public void Tick(float aTs)
        {
            base.Tick(aTs);

            System.Console.WriteLine($"{DateTime.UtcNow} --");
        }
    }
}