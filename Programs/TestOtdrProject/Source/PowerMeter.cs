using System;
using System.Linq;

using SpockEngine;

using Metrino.Otdr;
using Metrino.Otdr.Instrument;
using Metrino.Otdr.SignalProcessing;

using Metrino.Mono;

namespace Test
{
    public class TestPowerMeter : Script
    {

        private Instrument7000 mPowerMeter;

        private bool mPowerMeterStarted;

        private BlinkDetection mBlinkDetection;


        public TestPowerMeter() : base() { }

        override public void BeginScenario()
        {
            base.BeginScenario();

            var lConnectedModules = Instruments.GetConnectedModules();

            mPowerMeter = null;
            foreach (ModuleDescription lModule in lConnectedModules)
            {
                if (lModule.mName.StartsWith("OX1"))
                {
                    var lPowerMeter = lModule.Connect();

                    if (lPowerMeter.PowerMeterModeSupported)
                        mPowerMeter = lPowerMeter;
                }
            }

            mBlinkDetection = new BlinkDetection(new Interval(1.0, 0.1), new Interval(1.0, 0.1));
        }

        override public void EndScenario()
        {
            base.EndScenario();

            if (mPowerMeter != null)
                mPowerMeter.Stop();

            mPowerMeter = null;
            mBlinkDetection = null;
        }

        override public void Tick(float aTs)
        {
            base.Tick(aTs);

            if (mPowerMeter == null) return;

            if (!mPowerMeterStarted && (mPowerMeter.State == Metrino.Kernos.Instrument.State.Ready))
            {
                mPowerMeter.StartPowerMeterMode();
                mPowerMeterStarted = true;

                System.Console.WriteLine("Started power meter");
            }

            if (!mPowerMeterStarted)
                return;

            Metrino.Otdr.Value.Photocurrent lPowerValue = mPowerMeter.PowerValue;

            BlinkState lIsBlinking;
            mBlinkDetection.DetectHightestToneAndBlink(lPowerValue, out lIsBlinking);
            var valueLink = lPowerValue.Tag as Metrino.Otdr.PowerValue.ValueLink;

            System.Console.WriteLine($"{valueLink.Timestamp.Millisecond} -- {lPowerValue.Value} -- {lPowerValue.Power} -- {lIsBlinking}");
        }
    }
}