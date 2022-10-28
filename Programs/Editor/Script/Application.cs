using SpockEngine;

namespace SE
{
    public class TestApplication : SEApplication
    {
        public void Initialize()
        {
            base.Initialize(aTs);

        }

        public void Shutdown()
        {
            base.Shutdown(aTs);

        }

        public void Update(float aTs)
        {
            base.Update(aTs);
        }
    }
}