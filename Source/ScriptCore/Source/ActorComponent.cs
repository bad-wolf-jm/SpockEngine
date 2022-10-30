using SpockEngine;

namespace SpockEngine
{
    public class ActorComponent : Component
    {   
        private Entity mEntity;

        public ActorComponent() { mEntity = new Entity(); }
        public ActorComponent( Entity aEntity ) { mEntity = aEntity; }

        public virtual void OnCreate() {}

        public virtual void OnDestroy() {}

        public virtual void OnUpdate() {}
    }
}
