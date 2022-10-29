namespace SpockEngine
{
    public class Entity
    {
        private uint mEntityID;
        private ulong mRegistryID;

        public bool IsValid()
        {
            // return CppCall.IsValid( mRegistryID, mEntityID );

            return false;
        }

        public bool Has<_Component>() where _Component : Component, new()
        {
            // return CppCall.HasComponent( mRegistryID, mEntityID, typeof(_Component) );
            return false;
        }

        public void Tag<_Component>() where _Component : Component, new()
        {
            // CppCall.Tag( mRegistryID, mEntityID, typeof(_Component) );
        }

        public void Untag<_Component>() where _Component : Component, new()
        {
            // CppCall.Untag( mRegistryID, mEntityID, typeof(_Component) );
        }

        public void Add<_Component>(_Component aComponent) where _Component : Component, new()
        {
            if (Has<_Component>()) return;

            // CppCall.Add( mRegistryID, mEntityID, typeof(_Component), ref aComponent);
        }

        public void Replace<_Component>(_Component aComponent) where _Component : Component, new()
        {
            if (!Has<_Component>()) return;

            // CppCall.Replace( mRegistryID, mEntityID, typeof(_Component), ref aComponent);
        }

        public void AddOrReplace<_Component>(_Component aComponent) where _Component : Component, new()
        {
            // CppCall.AddOrReplace( mRegistryID, mEntityID, typeof(_Component), ref aComponent);
        }

        public void Remove<_Component>() where _Component : Component, new()
        {
            if (!Has<_Component>()) return;

            // CppCall.Replace( mRegistryID, mEntityID, typeof(_Component), ref aComponent);
        }

        public _Component Get<_Component>() where _Component : Component, new()
        {
            if (!Has<_Component>()) return new _Component();

            // CppCall.Replace( mRegistryID, mEntityID, typeof(_Component), out _Component lComponent);

            return new _Component();
        }
    }
}