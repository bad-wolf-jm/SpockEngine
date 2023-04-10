using SpockEngine;

namespace SpockEngine
{

    public interface IScript
    {
        void Begin();

        void End();

        bool Tick(float aTs);
    }
}
