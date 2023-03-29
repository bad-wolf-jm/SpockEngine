using SpockEngine;

namespace SpockEngine
{
    public class SEApplication
    {
        public SEApplication() { }

        public virtual void Initialize(string aConfigurationPath) { }

        public virtual void Shutdown(string aConfigurationPath) { }

        public virtual void Update(float aTs) { }

        public virtual void UpdateUI(float aTs) { }
    }
}
