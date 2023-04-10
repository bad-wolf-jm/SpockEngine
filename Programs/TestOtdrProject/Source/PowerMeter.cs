using System;
using System.Linq;

using SpockEngine;

using Metrino.Otdr;
using Metrino.Otdr.Instrument;
using Metrino.Otdr.SignalProcessing;

using Metrino.Interop;

namespace Test
{
    public class TestPowerMeter :  MarshalByRefObject, IScript
    {

        private Instrument7000 mPowerMeter;

        private bool mPowerMeterStarted;

        private BlinkDetection mBlinkDetection;

        private DateTime mStartTime;


        public TestPowerMeter() : base() { }

        public void Begin()
        {
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

            mBlinkDetection = new BlinkDetection();
            mStartTime = DateTime.Now;
        }

        public void End()
        {
            if (mPowerMeter != null)
            {
                mPowerMeter.Stop();
            }

            mPowerMeter = null;
            mBlinkDetection = null;
        }

        public bool Tick(float aTs)
        {
            if (mPowerMeter == null) return false;

            if (!mPowerMeterStarted && (mPowerMeter.State == Metrino.Kernos.Instrument.State.Ready))
            {
                mPowerMeter.StartPowerMeterMode();
                mPowerMeterStarted = true;

                System.Console.WriteLine("Started power meter");
            }

            if (!mPowerMeterStarted)
                return true;

            Metrino.Otdr.Value.Photocurrent lPowerValue = mPowerMeter.PowerValue;

            BlinkDetection.BlinkState lIsBlinking;
            var lFrequency = mBlinkDetection.DetectHightestToneAndBlink(lPowerValue, out lIsBlinking);

            var valueLink = lPowerValue.Tag as Metrino.Otdr.PowerValue.ValueLink;

            System.Console.WriteLine($"{(valueLink.Timestamp - mStartTime).TotalMilliseconds} -- {lPowerValue.Value} -- {lPowerValue.Power} -- {lFrequency.Value} -- {lIsBlinking}");

            return true;
        }
    }
}