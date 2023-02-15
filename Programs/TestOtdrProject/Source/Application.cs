using SpockEngine;
using System;

namespace Test
{
    public class TestScript : Script
    {
        public TestScript() : base() { }

        override public void BeginScenario()
        {
            base.BeginScenario();

            Console.WriteLine("Actor Component Created!!!");
        }

        override public void EndScenario()
        {
            base.EndScenario();

            Console.WriteLine("Actor Component Destroyed!!!");
        }

        override public void Tick(float aTs)
        {
            base.Tick(aTs);
        }
    }
}