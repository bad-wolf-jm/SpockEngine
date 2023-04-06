using SpockEngine;

namespace SpockEngine
{

    public interface IScript
    {
        void Begin();

        void End();

        void Tick(float aTs);
    }

    public class Script
    {
        public Script() { }

        public virtual void BeginScenario() { }

        public virtual void EndScenario() { }

        public virtual void Tick(float aTs) { }
    }
}
