using SpockEngine;

namespace SpockEngine
{
    public class ActorComponent
    {   
        protected Entity mEntity;

        public ActorComponent() { mEntity = new Entity(); }
        
        public void Initialize( Entity aEntity ) { mEntity = new Entity(aEntity); }

        public virtual void BeginScenario() {}

        public virtual void EndScenario() {}

        public virtual void Tick( float aTs ) {}
    }
}
