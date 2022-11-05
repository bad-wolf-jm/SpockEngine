using SpockEngine;
using SpockEngine.Math;
using System;

namespace Test
{
    public class TestActorComponent : ActorComponent
    {
        public TestActorComponent() : base() {}

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
                sNodeTransformComponent lTransform = mEntity.Get<sNodeTransformComponent>();
                mEntity.Replace<sNodeTransformComponent>(new sNodeTransformComponent(new mat4(3.0f)));
            }

            if (mEntity.Has<sTag>())
            {
                Console.WriteLine($"I have a tag");
                sTag lTransform = mEntity.Get<sTag>();
                mEntity.Replace<sTag>(new sTag("Sensor is running"));
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

            // Console.WriteLine($"Actor Component Updated {aTs}");
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