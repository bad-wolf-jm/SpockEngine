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
