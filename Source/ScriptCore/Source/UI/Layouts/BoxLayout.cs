using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public enum eBoxLayoutOrientation
    {
        HORIZONTAL,
        VERTICAL
    }

    public class UIBoxLayout : UIComponent
    {
        private List<UIComponent> mItems = new List<UIComponent>();
        private bool mDerived = false;
        public UIBoxLayout() : this(Interop.UIBoxLayout_Create(), false) { }

        public UIBoxLayout(IntPtr aInstance, bool aDerived) : base(aInstance) { mDerived = aDerived; }

        public UIBoxLayout(eBoxLayoutOrientation aOrientation) : this()
        {
            SetOrientation(aOrientation);
        }

        ~UIBoxLayout() { if (!mDerived) Interop.UIBoxLayout_Destroy(mInstance); }

        public void SetItemSpacing(float aItemSpacing)
        {
            Interop.UIBoxLayout_SetItemSpacing(mInstance, aItemSpacing);
        }

        public void Add(UIComponent aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            mItems.Add(aChild);
            var lInstance = (aChild != null) ? aChild.Instance : IntPtr.Zero;
            Interop.UIBoxLayout_AddAlignedNonFixed(mInstance, lInstance, aExpand, aFill, aHAlignment, aVAlignment);
        }

        public void Add(UIComponent aChild, bool aExpand, bool aFill)
        {
            mItems.Add(aChild);
            var lInstance = (aChild != null) ? aChild.Instance : IntPtr.Zero;
            Interop.UIBoxLayout_AddNonAlignedNonFixed(mInstance, lInstance, aExpand, aFill);
        }

        public void Add(UIComponent aChild, float aFixedSize, bool aExpand, bool aFill)
        {
            mItems.Add(aChild);
            var lInstance = (aChild != null) ? aChild.Instance : IntPtr.Zero;
            Interop.UIBoxLayout_AddNonAlignedFixed(mInstance, lInstance, aFixedSize, aExpand, aFill);
        }

        public void Add(UIComponent aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            mItems.Add(aChild);
            var lInstance = (aChild != null) ? aChild.Instance : IntPtr.Zero;
            Interop.UIBoxLayout_AddAlignedFixed(mInstance, lInstance, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment);
        }

        public void AddSeparator()
        {
            mItems.Add(null);
            Interop.UIBoxLayout_AddSeparator(mInstance);
        }

        public void Clear()
        {
            mItems.Clear();
            Interop.UIBoxLayout_Clear(mInstance);
        }
    }
}