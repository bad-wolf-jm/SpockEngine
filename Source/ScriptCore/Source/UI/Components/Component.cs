using System;
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
        protected IntPtr mInstance;
        public IntPtr Instance { get { return mInstance; } }

        public UIComponent() { mInstance = IntPtr.Zero; }
        public UIComponent(IntPtr aInstance) { mInstance = aInstance; }

        private bool mIsVisible;
        public bool IsVisible
        {
            get { return mIsVisible; }
            set { mIsVisible = value; Interop.UIComponent_SetIsVisible(mInstance, value); }
        }

        private bool mIsEnabled;
        public bool IsEnabled
        {
            get { return mIsEnabled; }
            set { mIsEnabled = value; Interop.UIComponent_SetIsEnabled(mInstance, value); }
        }

        private bool mAllowDragDrop;
        public bool AllowDragDrop
        {
            get { return mAllowDragDrop; }
            set { mAllowDragDrop = value; Interop.UIComponent_SetAllowDragDrop(mInstance, value); }
        }

        private UIComponent mTooltip;
        public UIComponent Tooltip { set { mTooltip = value; Interop.UIComponent_SetTooltip(mInstance, mTooltip.Instance); } }

        public void SetPadding(float aPaddingAll)
        {
            Interop.UIComponent_SetPaddingAll(mInstance, aPaddingAll);
        }

        public void SetPadding(float aPaddingTopBottom, float aPaddingLeftRight)
        {
            Interop.UIComponent_SetPaddingPairs(mInstance, aPaddingTopBottom, aPaddingLeftRight);
        }

        public void SetPadding(float aPaddingTop, float aPaddingBottom, float aPaddingLeft, float aPaddingRight)
        {
            Interop.UIComponent_SetPaddingIndividual(mInstance, aPaddingTop, aPaddingBottom, aPaddingLeft, aPaddingRight);
        }

        public void SetAlignment(eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            Interop.UIComponent_SetAlignment(mInstance, aHAlignment, aVAlignment);
        }

        public void SetHorizontalAlignment(eHorizontalAlignment aAlignment)
        {
            Interop.UIComponent_SetHorizontalAlignment(mInstance, aAlignment);
        }

        public void SetVerticalAlignment(eVerticalAlignment aAlignment)
        {
            Interop.UIComponent_SetVerticalAlignment(mInstance, aAlignment);
        }

        public void SetBackgroundColor(Math.vec4 aColor)
        {
            Interop.UIComponent_SetBackgroundColor(mInstance, aColor);
        }

        public void SetFont(eFontFamily aFont)
        {
            Interop.UIComponent_SetFont(mInstance, aFont);
        }
    }
}
