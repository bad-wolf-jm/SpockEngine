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
            mTestField0 = 0.0f;
        }

        override public void OnDestroy()
        {
            base.OnDestroy();

            Console.WriteLine("Actor Component Destroyed!!!");
        }

        override public void OnUpdate(float aTs )
        {
            base.OnUpdate(aTs);
          
            if (mEntity.Has<sNodeTransformComponent>())
            {
                sNodeTransformComponent lTransform = mEntity.Get<sNodeTransformComponent>();
                mat4 lDeltaRotation = mat4.Rotation(3.1415f * mTestField0 / 300.0f, new vec3(0.0f, 1.0f, 0.0f));
                mTestField0 += aTs;

                mEntity.Replace<sNodeTransformComponent>(new sNodeTransformComponent(lDeltaRotation));
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