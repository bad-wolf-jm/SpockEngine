using SpockEngine;
using System;

namespace Test
{
    public class TestActorComponent : ActorComponent
    {
        override public void OnCreate() 
        {
            base.OnCreate();

            Console.WriteLine("Actor Component Created!!!");
        }

        override public void OnDestroy()
        {
            base.OnDestroy();

            Console.WriteLine("Actor Component Destroyed!!!");
        }

        override public void OnUpdate()
        {
            base.OnUpdate();
            
            Console.WriteLine("Actor Component Updated!!!");
        }
    }

    public class TestApplication : SEApplication
    {
        override public void BeginScenario() 
        {
            base.BeginScenario();
        }

        override public void EndScenario()
        {
            base.EndScenario();
        }

        override public void Update(float aTs)
        {
            base.Update(aTs);
        }
    }
}