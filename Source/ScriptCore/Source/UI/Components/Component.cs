using System.Runtime.CompilerServices;

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

        public void SetPadding(float aPaddingAll)
        {
            UIComponent_SetPaddingAll(mInstance, aPaddingAll);
        }

        public void SetPadding(float aPaddingTopBottom, float aPaddingLeftRight)
        {
            UIComponent_SetPaddingPairs(mInstance, aPaddingTopBottom, aPaddingLeftRight);
        }

        public void SetPadding(float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight)
        {
            UIComponent_SetPaddingIndividual(mInstance, aPaddingTop, aPaddingBottom, aPaddingLeft, aPaddingRight);
        }

        public void SetAlignment(eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            UIComponent_SetAlignment(mInstance, aHAlignment, aVAlignment);
        }

        public void SetHorizontalAlignment(eHorizontalAlignment aAlignment)
        {
            UIComponent_SetHorizontalAlignment(mInstance, aAlignment);
        }

        public void SetVerticalAlignment(eVerticalAlignment aAlignment)
        {
            UIComponent_SetVerticalAlignment(mInstance, aAlignment);
        }

        public void SetBackgroundColor(Math.vec4 aColor)
        {
            UIComponent_SetBackgroundColor(mInstance, aColor);
        }

        public void SetFont(eFontFamily aFont)
        {
            UIComponent_SetFont(mInstance, aFont);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetPaddingAll(ulong aSelf, float aPaddingAll);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetPaddingPairs(ulong aSelf, float aPaddingTopBottom, float aPaddingLeftRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetPaddingIndividual(ulong aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetAlignment(ulong aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetHorizontalAlignment(ulong aSelf, eHorizontalAlignment aAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetVerticalAlignment(ulong aSelf, eVerticalAlignment aAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetBackgroundColor(ulong aSelf, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComponent_SetFont(ulong aSelf, eFontFamily aFont);
    }
}
