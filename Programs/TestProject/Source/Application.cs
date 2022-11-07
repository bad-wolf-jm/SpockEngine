using SpockEngine;
using SpockEngine.Math;
using System;

namespace Test
{
    public class TestActorComponent : ActorComponent
    {
        public float mTestField0;
        private float mTestField2;

        public TestActorComponent() : base() {}

        override public void OnCreate() 
        {
            base.OnCreate();

            Console.WriteLine($"Actor Component Created");

            // if (mEntity.IsValid())
            // {
            //     Console.WriteLine($"Valid entity");
            // }

            // if (mEntity.Has<sNodeTransformComponent>())
            // {
            //     Console.WriteLine($"I have a transform");
            //     sNodeTransformComponent lTransform = mEntity.Get<sNodeTransformComponent>();
            //     mEntity.Replace<sNodeTransformComponent>(new sNodeTransformComponent(new mat4(3.0f)));
            // }

            // if (mEntity.Has<sTag>())
            // {
            //     Console.WriteLine($"I have a tag");
            //     sTag lTransform = mEntity.Get<sTag>();
            //     mEntity.Replace<sTag>(new sTag("Sensor is running"));
            // }
        }

        override public void OnDestroy()
        {
            base.OnDestroy();

            Console.WriteLine("Actor Component Destroyed!!!");
        }

        override public void OnUpdate(float aTs )
        {
            base.OnUpdate(aTs);
            
            if (mEntity.IsValid())
            {
                Console.WriteLine($"Valid entity");
            }

            // Console.WriteLine(mEntity.ToString());
            if (mEntity.Has<sNodeTransformComponent>())
            {
            // // //     sNodeTransformComponent lTransform = mEntity.Get<sNodeTransformComponent>();
            // // //     mat4 lDeltaRotation = mat4.Rotation(1.07f / 180.0f, new vec3(0.0f, 1.0f, 0.0f));

            // // //     // mEntity.Replace<sNodeTransformComponent>(new sNodeTransformComponent(lDeltaRotation * lTransform.mMatrix));
            }

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