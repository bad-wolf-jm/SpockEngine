using System;

using SpockEngine;

using Metrino.Otdr.Instrument;

using Metrino.Mono;

namespace Test
{
    public class TestScript : Script
    {

        private Instrument7000 mSource;
        private Instrument7000 mPowerMeter;

        private bool mSourceStarted;
        private bool mPowerMeterStarted;


        public TestScript() : base() { }

        override public void BeginScenario()
        {
            base.BeginScenario();

            var lConnectedModules = Instruments.GetConnectedModules();

            mSource = null;
            mPowerMeter = null;
            foreach (ModuleDescription lModule in lConnectedModules)
            {
                if (lModule.mName.StartsWith("FTB"))
                {
                    mSource = lModule.Connect();
                }

                if (lModule.mName.StartsWith("OX1"))
                {
                    var lPowerMeter = lModule.Connect();

                    if (lPowerMeter.PowerMeterModeSupported)
                        mPowerMeter = lPowerMeter;
                }
            }
        }

        override public void EndScenario()
        {
            base.EndScenario();
        }

        override public void Tick(float aTs)
        {
            base.Tick(aTs);

            if ((mSource == null) || (mPowerMeter == null)) return;

            if (!mSourceStarted && (mSource.State == Metrino.Kernos.Instrument.State.Ready))
            {
                Configuration7000 lSourceConfiguration = new Configuration7000();
                lSourceConfiguration.SourceWavelength = 0.0;
                lSourceConfiguration.SourceFrequency = 0.0;
                lSourceConfiguration.SourceModulation = true;
                lSourceConfiguration.SourceBlinkModulation = true;

                mSource.StartSourceMode(lSourceConfiguration);
                mSourceStarted = true;

                System.Console.WriteLine("Started power source");
            }

            if (!mPowerMeterStarted && (mPowerMeter.State == Metrino.Kernos.Instrument.State.Ready))
            {
                mPowerMeter.StartPowerMeterMode();
                mPowerMeterStarted = true;

                System.Console.WriteLine("Started power meter");
            }

            if (!mSourceStarted || !mPowerMeterStarted)
                return;

            Metrino.Otdr.Value.Photocurrent lPowerValue = mPowerMeter.PowerValue;
            System.Console.WriteLine(lPowerValue);
        }
    }
}