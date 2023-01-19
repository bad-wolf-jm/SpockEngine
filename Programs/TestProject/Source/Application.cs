using SpockEngine;
using SpockEngine.Math;
using System;

namespace Test
{
    public class TestActorComponent : ActorComponent
    {
        public float mTestField0;
        // private float mTestField2;

        public TestActorComponent() : base() {}

        override public void BeginScenario() 
        {
            base.BeginScenario();
            mTestField0 = 0.0f;

            base.CreateEntity("HELLO!!!");
        }

        override public void EndScenario()
        {
            base.EndScenario();

            Console.WriteLine("Actor Component Destroyed!!!");
        }

        override public void Tick(float aTs )
        {
            base.Tick(aTs);
          
            if (mEntity.Has<sNodeTransformComponent>())
            {
                sNodeTransformComponent lTransform = mEntity.Get<sNodeTransformComponent>();
                mat4 lDeltaRotation = mat4.Rotation(3.1415f * mTestField0 / 300.0f, new vec3(0.0f, 1.0f, 0.0f));
                mTestField0 += aTs;

                mEntity.Replace<sNodeTransformComponent>(new sNodeTransformComponent(lDeltaRotation));
            }

        }
    }

    public class TestHUDComponent : HUDComponent
    {
        public float mTestField0;
        public bool  mButtonClicked;

        public TestHUDComponent() : base() { mButtonClicked = false; }

        override public void BeginScenario() 
        {
            base.BeginScenario();
            mTestField0 = 0.0f;
            mButtonClicked = false;
            Console.WriteLine("HUD Component Destroyed!!!");
        }

        override public void EndScenario()
        {
            base.EndScenario();

            Console.WriteLine("HUD Component Destroyed!!!");
        }

        override public void DrawContent(float aTs )
        {
            base.DrawContent(aTs);

            UI.Text("HUD element draws text");          
            if (UI.Button("Button"))
            {
                mButtonClicked = !mButtonClicked;
            }
            if (mButtonClicked) UI.Text("HUD element draws text after button clicked");          

        }

        override public void DrawPreviewContent(float aTs )
        {
            base.DrawContent(aTs);

            UI.Text("HUD element draws previewtext");          

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