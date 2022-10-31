using SpockEngine;
using System;

namespace Test
{
    public class TestActorComponent : ActorComponent
    {
        public TestActorComponent() : base() {}
        // public TestActorComponent(Entity e) : base(e) 
        // {
        // }

        override public void OnCreate() 
        {
            base.OnCreate();

            Console.WriteLine($"Actor Component Created entity");

            if (mEntity.IsValid())
            {
                Console.WriteLine($"Valid entity");
            }

            if (mEntity.Has<sNodeTransformComponent>())
            {
                Console.WriteLine($"I have a transform");
            }

        }

        override public void OnDestroy()
        {
            base.OnDestroy();

            Console.WriteLine("Actor Component Destroyed!!!");
        }

        override public void OnUpdate(float aTs )
        {
            base.OnUpdate(aTs);

            Console.WriteLine($"Actor Component Updated {aTs}");
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