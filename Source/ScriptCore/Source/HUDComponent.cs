using SpockEngine;

namespace SpockEngine
{
    public class HUDComponent
    {
        protected Entity mEntity;

        public HUDComponent() { mEntity = new Entity(); }

        public void Initialize(Entity aEntity) { mEntity = new Entity(aEntity); }

        public virtual void BeginScenario() { }

        public virtual void EndScenario() { }

        public virtual void DrawContent(float aTs) { }
    }
}
