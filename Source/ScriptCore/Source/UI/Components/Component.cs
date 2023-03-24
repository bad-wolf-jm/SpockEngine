namespace SpockEngine
{
    public enum eHorizontalAlignment
    {
        LEFT,
        RIGHT,
        CENTER
    }

    public enum eVerticalAlignment
    {
        TOP,
        BOTTOM,
        CENTER
    }

    public abstract class UIComponent
    {
        protected ulong mInstance;
        public ulong Instance { get { return mInstance; } }

        public UIComponent() { mInstance = 0; }
        public UIComponent(ulong aInstance) { mInstance = aInstance; }

        // public abstract void Update();        
    }
}
