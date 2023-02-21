using System;
using System.Linq;

using SpockEngine;

using Metrino.Otdr;
using Metrino.Otdr.Instrument;
using Metrino.Otdr.SignalProcessing;

using Metrino.Mono;

namespace Test
{
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

            mBlinkDetection = new BlinkDetection();
        }

        override public void EndScenario()
        {
            base.EndScenario();

            if (mSource != null)
                mSource.Stop();
                
            if (mPowerMeter != null)
                mPowerMeter.Stop();

            mSource = null;
            mPowerMeter = null;
            mBlinkDetection = null;
        }

        override public void Tick(float aTs)
        {
            base.Tick(aTs);

            if ((mSource == null) || (mPowerMeter == null)) return;

            if (!mSourceStarted && (mSource.State == Metrino.Kernos.Instrument.State.Ready))
            {
                Configuration7000 lSourceConfiguration = (Configuration7000)mSource.OtdrCurrentConfiguration.Clone();

                foreach(var lWL in mSource.SettingsValidator.Wavelengths)
                {
                    if (PhysicalFiberCharacteristics.IsMonoModeFiber(lWL.FiberCode))
                    {
                        lSourceConfiguration.SourceWavelength = lWL.Wavelength;
                        break;
                    }
                }

                lSourceConfiguration.SourceModulation = true;
                lSourceConfiguration.SourceModulationValue = 1000;
                lSourceConfiguration.SourceBlinkModulation = false;
                lSourceConfiguration.SourceBlinkModulationValue = 1;

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

            BlinkState lIsBlinking;
            var lFrequency = mBlinkDetection.DetectHightestToneAndBlink(lPowerValue, out lIsBlinking);
            var valueLink = lPowerValue.Tag as Metrino.Otdr.PowerValue.ValueLink;

            // System.Console.WriteLine($"{valueLink.Timestamp} -- {lPowerValue.Value} -- {lPowerValue.Power} -- {lFrequency.Value} -- {lIsBlinking}");
        }
    }
}