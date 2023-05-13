using SpockEngine;

namespace SpockEngine
{
    public class Script
    {
        public Script() { }

        public virtual void BeginScenario() { }

        public virtual void EndScenario() { }

        public virtual void Tick(float aTs) { }
    }
}
