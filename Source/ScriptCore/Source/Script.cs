using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

using SpockEngine;

namespace SpockEngine
{

    public interface IScript
    {
        void Begin();

        void End();

        bool Tick(float aTs);
    }

    public class RemoteScript : MarshalByRefObject, IScript
    {
        object mScriptObject;

        public RemoteScript(object aScriptObject) { mScriptObject = aScriptObject; }

        public void Begin()
        {
            if (mScriptObject == null) return;

            mScriptObject.GetType().GetMethod("Begin").Invoke(mScriptObject, null);
        }

        public void End()
        {
            if (mScriptObject == null) return;

            mScriptObject.GetType().GetMethod("End").Invoke(mScriptObject, null);
        }

        public bool Tick(float aTs)
        {
            if (mScriptObject == null) return false;

            return (bool)mScriptObject.GetType().GetMethod("Tick").Invoke(mScriptObject, new object[] { aTs });
        }
    }


    public class Script : MarshalByRefObject, IScript
    {
        public Script() { }

        public virtual void Begin()
        {
        }

        public virtual void End()
        {
        }

        public virtual bool Tick(float aTs)
        {
            return true;
        }
    }


}
