namespace SpockEngine
{
    public abstract class UIComponent
    {
        protected ulong mInstance;
        public ulong Instance { get { return mInstance; } }

        public UIComponent() { mInstance = 0; }
        public UIComponent(ulong aInstance) { mInstance = aInstance; }

        public abstract void Update();        
    }
}
