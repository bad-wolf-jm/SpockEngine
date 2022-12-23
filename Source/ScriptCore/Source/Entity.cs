using System;

namespace SpockEngine
{
    public class Entity
    {
        private uint mEntityID;
        private ulong mRegistryID;

        public Entity() { mEntityID = 0; mRegistryID = 0; }
        public Entity(Entity aOther) { mEntityID = aOther.mEntityID; mRegistryID = aOther.mRegistryID; }
        public Entity(uint aEntityID, ulong aRegistryID) { mEntityID = aEntityID; mRegistryID = aRegistryID; }

        public Entity CreateEntity(string aName)
        {
            uint lNewEntityID = CppCall.Entity_Create(mRegistryID, aName, mEntityID);

            return new Entity(lNewEntityID, mRegistryID);
        }

        public override string ToString()
        {
            return $"Entity(mEntityID={mEntityID} -- mRegistryID={mRegistryID})";
        }

        public bool IsValid()
        {
            if (mRegistryID == 0) return false;
            
            return CppCall.Entity_IsValid(mEntityID, mRegistryID);
        }

        public bool Has<_Component>() where _Component : Component, new()
        {
            return CppCall.Entity_Has(mEntityID, mRegistryID, typeof(_Component));
        }

        public _Component Get<_Component>() where _Component : Component, new()
        {
            if (!Has<_Component>()) return new _Component();

            return CppCall.Entity_Get<_Component>(mEntityID, mRegistryID, typeof(_Component));
        }

        public void Add<_Component>(_Component aComponent) where _Component : Component, new()
        {
            if (Has<_Component>()) return;

            CppCall.Entity_Add<_Component>(mEntityID, mRegistryID, typeof(_Component), aComponent);
        }

        public void Replace<_Component>(_Component aComponent) where _Component : Component, new()
        {
            if (!Has<_Component>()) return;

            CppCall.Entity_Replace<_Component>(mEntityID, mRegistryID, typeof(_Component), aComponent);
        }

        public void Remove<_Component>() where _Component : Component, new()
        {
            if (!Has<_Component>()) return;

            CppCall.Entity_Remove(mEntityID, mRegistryID, typeof(_Component));
        }
    }
}