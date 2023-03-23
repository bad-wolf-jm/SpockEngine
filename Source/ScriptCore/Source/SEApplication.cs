using SpockEngine;

namespace SpockEngine
{
    public class SEApplication
    {
        public SEApplication() { }

        public virtual void Initialize() { }

        public virtual void Shutdown() { }

        public virtual void Update(float aTs) { }

        public virtual void UpdateUI(float aTs) { }
    }
}
