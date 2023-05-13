using System;
using System.Runtime.CompilerServices;

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
            uint lNewEntityID = Entity_Create(mRegistryID, aName, mEntityID);

            return new Entity(lNewEntityID, mRegistryID);
        }

        public override string ToString()
        {
            return $"Entity(mEntityID={mEntityID} -- mRegistryID={mRegistryID})";
        }

        public bool IsValid()
        {
            if (mRegistryID == 0) return false;

            return Entity_IsValid(mEntityID, mRegistryID);
        }

        public bool Has<_Component>() where _Component : Component, new()
        {
            return Entity_Has(mEntityID, mRegistryID, typeof(_Component));
        }

        public _Component Get<_Component>() where _Component : Component, new()
        {
            if (!Has<_Component>()) return new _Component();

            return Entity_Get<_Component>(mEntityID, mRegistryID, typeof(_Component));
        }

        public void Add<_Component>(_Component aComponent) where _Component : Component, new()
        {
            if (Has<_Component>()) return;

            Entity_Add<_Component>(mEntityID, mRegistryID, typeof(_Component), aComponent);
        }

        public void Replace<_Component>(_Component aComponent) where _Component : Component, new()
        {
            if (!Has<_Component>()) return;

            Entity_Replace<_Component>(mEntityID, mRegistryID, typeof(_Component), aComponent);
        }

        public void Remove<_Component>() where _Component : Component, new()
        {
            if (!Has<_Component>()) return;

            Entity_Remove(mEntityID, mRegistryID, typeof(_Component));
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint Entity_Create(ulong aRegistry, string aName, uint aParentEntityID);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static bool Entity_IsValid(uint aEntityID, ulong aRegistry);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static bool Entity_Has(uint aEntityID, ulong aRegistry, Type aTypeDesc);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static T Entity_Get<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc) where T : Component, new();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void Entity_Add<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc, T aNewValue) where T : Component, new();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void Entity_Replace<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc, T aNewValue) where T : Component, new();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void Entity_Remove(uint aEntityID, ulong aRegistry, Type aTypeDesc);
    }
}